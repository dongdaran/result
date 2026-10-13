"""Single-prompt Hugging Face inference and decoder-output activations.

Heavy dependencies are imported only when creating a runner. Downloading and
splitting the dataset therefore works without PyTorch, CUDA, or Transformers.
"""

from contextlib import ExitStack, nullcontext

from .steering import block_hidden, resolve_layer_index, steering_hook


def last_token_positions(attention_mask):
    """Find the final nonpadding token for left- or right-padded batches."""
    import torch

    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must have nonempty shape [batch, tokens].")
    occupied = attention_mask != 0
    if not occupied.any(dim=1).all().item():
        raise ValueError("Cannot select a token from an empty or entirely padded prompt.")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.expand_as(attention_mask).masked_fill(~occupied, -1).max(dim=1).values


def decoder_blocks(model):
    """Explicitly support Llama/Qwen/Mistral decoder stacks and GPT-style h."""
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        transformer = getattr(model, "transformer", None)
        layers = getattr(transformer, "h", None)
    if layers is None or len(layers) == 0:
        raise ValueError(
            f"Unsupported decoder stack for {type(model).__name__}. "
            "Expected model.model.layers or model.transformer.h; add an explicit adapter."
        )
    return layers


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class HFRunner:
    """Load one immutable model revision onto one device; process explicitly configured prompt batches.

    ``activation_means`` and ``generate`` share the exact tokenizer/chat path.
    Activations are decoder BLOCK OUTPUT (not post-final-norm hidden states),
    sampled at the last nonpadding prompt token and averaged in float32 on CPU.
    """

    def __init__(self, model_config: dict, runtime_config: dict):
        self.hf_id = str(model_config.get("hf_id") or "").strip()
        self.revision = str(model_config.get("revision") or "").strip()
        if not self.hf_id or not self.revision:
            raise ValueError("Set a nonempty hf_id and pinned revision before loading a model.")
        self.batch_size = _positive_int(runtime_config.get("batch_size", 1), "batch_size")
        self.max_input_tokens = _positive_int(
            runtime_config.get("max_input_tokens", 2048), "max_input_tokens"
        )
        self.max_new_tokens = _positive_int(
            runtime_config.get("max_new_tokens", 256), "max_new_tokens"
        )
        self.do_sample = runtime_config.get("do_sample", False)
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be a boolean.")
        self.seed = runtime_config.get("seed", 42)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        dtype_name = model_config.get("dtype", "auto")
        if dtype_name not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16.")
        device_name = str(runtime_config.get("device", "cuda"))
        if (
            device_name != "cpu"
            and device_name != "cuda"
            and not (device_name.startswith("cuda:") and device_name[5:].isdigit())
        ):
            raise ValueError("device must be cpu, cuda, or cuda:N.")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model execution needs GPU dependencies: uv sync --extra gpu"
            ) from exc
        self._torch = torch
        self.device = torch.device(device_name)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable. Use a CUDA Runpod image or device=cpu."
                )
            device_index = (
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
            if device_index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_index} does not exist.")
            with torch.cuda.device(device_index):
                bf16_supported = torch.cuda.is_bf16_supported()
            if dtype_name == "bfloat16" and not bf16_supported:
                raise RuntimeError(
                    "The selected CUDA GPU does not support bfloat16; select float16 or float32."
                )
            if dtype_name == "auto":
                dtype_name = "bfloat16" if bf16_supported else "float16"
        elif dtype_name == "auto":
            dtype_name = "float32"
        self.dtype_name = dtype_name
        self.dtype = getattr(torch, dtype_name)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.requested_revision = self.revision
        config = AutoConfig.from_pretrained(
            self.hf_id, revision=self.revision, trust_remote_code=False
        )
        self.revision = getattr(config, "_commit_hash", None) or self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, revision=self.revision, trust_remote_code=False
        )
        chat_setting = model_config.get("use_chat_template", True)
        if chat_setting != "auto" and not isinstance(chat_setting, bool):
            raise ValueError("use_chat_template must be auto, true, or false.")
        self.use_chat_template = True if chat_setting == "auto" else chat_setting
        if self.use_chat_template and not self.tokenizer.chat_template:
            raise ValueError("use_chat_template=true but this tokenizer has no chat template.")
        self.system_prompt = model_config.get("system_prompt")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string or null.")
        if self.system_prompt and not self.use_chat_template:
            raise ValueError("system_prompt requires use_chat_template=true.")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                revision=self.revision,
                config=config,
                torch_dtype=self.dtype,
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
        )
        self.layers = decoder_blocks(self.model)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "n_embd", None
        )
        self.hidden_size = _positive_int(hidden_size, "model hidden_size")
        self._context_window = getattr(self.model.config, "max_position_embeddings", None)
        self._closed = False

    def _encode_prompt(self, prompt: str):
        return self._encode_prompts([prompt])

    def _encode_prompts(self, prompts: list[str]):
        if self._closed:
            raise RuntimeError("HFRunner has been closed.")
        if not prompts or len(prompts) > self.batch_size:
            raise ValueError("Prompt batch must be nonempty and no larger than batch_size.")
        texts = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a nonempty string.")
            text = prompt
            if self.use_chat_template:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad_token_id or eos_token_id for batching.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            texts, return_tensors="pt", add_special_tokens=not self.use_chat_template,
            truncation=False, padding=True, return_attention_mask=True,
            return_token_type_ids=False,
        )
        token_count = inputs["input_ids"].shape[1]
        if token_count > self.max_input_tokens:
            raise ValueError(
                f"Prompt contains {token_count} tokens, above max_input_tokens={self.max_input_tokens}; no truncation performed."
            )
        if isinstance(self._context_window, int) and token_count > self._context_window:
            raise ValueError(
                f"Prompt length {token_count} exceeds model context window {self._context_window}."
            )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def activation_means(self, prompts: list[str], layers: list[int]) -> dict:
        torch = self._torch
        if not prompts:
            raise ValueError("Activation extraction needs at least one prompt.")
        selected = [resolve_layer_index(index, len(self.layers)) for index in layers]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(
                "Select at least one layer; duplicate resolved layers are not allowed."
            )
        totals = {index: torch.zeros(self.hidden_size, dtype=torch.float32) for index in selected}
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            inputs = self._encode_prompts(batch)
            position_ids = inputs["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(inputs["attention_mask"] == 0, 0)
            positions = last_token_positions(inputs["attention_mask"])
            captured = {}

            def capture(index):
                def hook(_module, _args, output):
                    hidden = block_hidden(output)
                    if hidden.shape[-1] != self.hidden_size:
                        raise ValueError("Observed decoder width does not match model hidden_size.")
                    batch_ids = torch.arange(hidden.shape[0], device=hidden.device)
                    values = (
                        hidden[batch_ids, positions.to(hidden.device)]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                    )
                    if (
                        values.shape != (len(batch), self.hidden_size)
                        or not torch.isfinite(values).all().item()
                    ):
                        raise ValueError("Invalid or nonfinite extracted prompt activation.")
                    captured[index] = values

                return hook

            with ExitStack() as stack:
                for index in selected:
                    handle = self.layers[index].register_forward_hook(capture(index))
                    stack.callback(handle.remove)
                with torch.inference_mode():
                    self.model(**inputs, position_ids=position_ids, use_cache=False)
            if set(captured) != set(selected):
                raise RuntimeError("One or more requested decoder blocks were not called.")
            for index in selected:
                for value in captured[index]:
                    totals[index] += value
        return {index: value / len(prompts) for index, value in totals.items()}

    def generate(self, prompt: str, steering: dict | None = None) -> str:
        return self.generate_batch([prompt], steering)[0]

    def generate_batch(self, prompts: list[str], steering: dict | None = None) -> list[str]:
        torch = self._torch
        inputs = self._encode_prompts(prompts)
        prompt_length = inputs["input_ids"].shape[1]
        if (
            isinstance(self._context_window, int)
            and prompt_length + self.max_new_tokens > self._context_window
        ):
            raise ValueError(
                "Prompt plus max_new_tokens exceeds the model context window; reduce a configured limit."
            )
        context = nullcontext()
        if steering is not None:
            required = {"layer", "vector", "coefficient"}
            if not required.issubset(steering) or set(steering) - required - {"normalize"}:
                raise ValueError(
                    "Steering requires layer, vector, coefficient, and optional normalize."
                )
            if not isinstance(steering["vector"], torch.Tensor) or steering["vector"].ndim != 1:
                raise ValueError("Steering vector must be a one-dimensional torch.Tensor.")
            if steering["vector"].numel() != self.hidden_size:
                raise ValueError("Steering vector does not match this model's hidden_size.")
            context = steering_hook(
                self.layers,
                steering["layer"],
                steering["vector"],
                steering["coefficient"],
                normalize=steering.get("normalize", False),
            )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id for generation.")
        # Each sampled comparison starts from the same RNG state.
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        with context, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=pad_id,
                use_cache=True,
            )
        if output.ndim != 2 or output.shape[0] != len(prompts):
            raise ValueError("Generation output does not match prompt batch.")
        return self.tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)

    def close(self):
        """Release model tensors before loading another model or the judge."""
        self.model = None
        self.layers = None
        self.tokenizer = None
        self._closed = True
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()
