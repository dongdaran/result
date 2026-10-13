"""Engine regression tests run on CPU with random weights, never downloaded models."""

import ast
from pathlib import Path

import pytest

from hb_steering.evaluate import (
    LLAMA2_CLS_PROMPT,
    JudgeOutputError,
    format_judge_prompt,
    parse_judge_label,
)
from hb_steering.models import HFRunner, last_token_positions
from hb_steering.steering import resolve_layer_index, steering_hook
from hb_steering.vectors import compute_vectors, load_vectors, save_vectors


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def runner(torch, monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {
        token: index
        for index, token in enumerate(
            (
                "<pad>",
                "<bos>",
                "<eos>",
                "<unk>",
                "user",
                "assistant",
                "hello",
                "safe",
                "unsafe",
                "test",
            )
        )
    }
    tokenizer_core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "<bos> {% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    config = transformers.LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._commit_hash = "a" * 40
    torch.manual_seed(7)
    model = transformers.LlamaForCausalLM(config).eval()
    loads = []

    def load_config(*args, **kwargs):
        loads.append(("config", kwargs["revision"]))
        return config

    def load_tokenizer(*args, **kwargs):
        loads.append(("tokenizer", kwargs["revision"]))
        return tokenizer

    def load_model(*args, **kwargs):
        loads.append(("model", kwargs["revision"]))
        return model

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    instance = HFRunner(
        {"hf_id": "local-test", "revision": "main", "dtype": "float32"},
        {"device": "cpu", "max_input_tokens": 32, "max_new_tokens": 2},
    )
    assert loads == [("config", "main"), ("tokenizer", "a" * 40), ("model", "a" * 40)]
    yield instance
    instance.close()


def test_last_nonpadding_token_handles_both_padding_sides(torch):
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])
    assert last_token_positions(mask).tolist() == [3, 1, 2]
    with pytest.raises(ValueError, match="entirely padded"):
        last_token_positions(torch.zeros((1, 2), dtype=torch.long))


def test_extraction_uses_block_output_before_final_norm(runner, torch):
    inputs = runner._encode_prompt("hello safe")
    captured = []
    handle = runner.layers[-1].register_forward_hook(
        lambda _module, _args, out: captured.append(
            out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()
        )
    )
    try:
        with torch.inference_mode():
            result = runner.model(**inputs, use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    expected = captured[0][0, -1].float().cpu()
    actual = runner.activation_means(["hello safe"], [-1])[1]
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, result.hidden_states[-1][0, -1].float().cpu())
    assert all(not block._forward_hooks for block in runner.layers)


def test_extraction_mean_matches_individual_prompt_mean(runner, torch):
    prompts = ["hello safe", "hello unsafe"]
    together = runner.activation_means(prompts, [0, -1])
    singles = [runner.activation_means([prompt], [0, -1]) for prompt in prompts]
    for layer in together:
        torch.testing.assert_close(together[layer], (singles[0][layer] + singles[1][layer]) / 2)
    with pytest.raises(ValueError, match="duplicate"):
        runner.activation_means(prompts, [1, -1])


def test_steering_preserves_tuple_and_affects_prompt_and_generation_tokens(torch):
    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 2, "cache-sentinel"

    block = Block()
    vector = torch.tensor([1.0, -2.0, 3.0])
    layers = [block]
    with steering_hook(layers, -1, vector, -0.5):
        for length in (4, 1):
            inputs = torch.ones(1, length, 3)
            output, cache = block(inputs)
            torch.testing.assert_close(output, inputs * 2 - 0.5 * vector)
            assert cache == "cache-sentinel"
    assert not block._forward_hooks
    with steering_hook(layers, 0, vector, 2, normalize=True):
        output, _cache = block(torch.zeros(1, 1, 3))
        torch.testing.assert_close(output[0, 0], 2 * vector / vector.norm())


def test_steering_zero_is_exact_and_hooks_cleanup_on_exception(torch):
    block = torch.nn.Identity()
    inputs = torch.randn(2, 3, 4)
    with steering_hook([block], 0, torch.ones(4), 0):
        assert block(inputs) is inputs
        assert not block._forward_hooks
    with pytest.raises(RuntimeError, match="intentional"):
        with steering_hook([block], 0, torch.ones(4), 1):
            raise RuntimeError("intentional")
    assert not block._forward_hooks
    with pytest.raises(ValueError, match="width"):
        with steering_hook([block], 0, torch.ones(2), 1):
            block(inputs)
    assert not block._forward_hooks


def test_runner_baseline_and_zero_intervention_are_identical(runner, torch):
    baseline = runner.generate("hello safe")
    steered = runner.generate(
        "hello safe", {"layer": -1, "vector": torch.ones(8), "coefficient": 0.0}
    )
    assert baseline == steered
    with pytest.raises(ValueError, match="no truncation"):
        runner.generate(" ".join(["hello"] * 40))
    assert not runner.layers[-1]._forward_hooks


def test_tiny_llama_vectors_feed_the_actual_generation_hook(runner, torch):
    payload = compute_vectors(
        runner,
        {1: ["hello unsafe"], 2: ["hello test"], 3: ["unsafe test"]},
        ["hello safe", "safe test"],
        [-1],
    )
    vector = payload["vectors"][1][1]
    assert vector.shape == (runner.hidden_size,)
    assert torch.isfinite(vector).all().item()
    assert vector.norm().item() > 0
    baseline = runner.generate("hello test")
    intervention = {"layer": 1, "vector": vector, "coefficient": 0.0}
    assert runner.generate("hello test", intervention) == baseline
    intervention["coefficient"] = 0.3
    assert isinstance(runner.generate("hello test", intervention), str)
    assert all(not block._forward_hooks for block in runner.layers)


def test_difference_vectors_and_safe_round_trip(torch, tmp_path):
    class FakeRunner:
        layers = [None, None]
        hidden_size = 2
        hf_id = "example/model"
        revision = "a" * 40

        def activation_means(self, prompts, layers):
            means = {"benign": [1.0, 3.0], "low": [2.0, 1.0], "high": [4.0, 7.0]}
            return {layer: torch.tensor(means[prompts[0]]) for layer in layers}

    payload = compute_vectors(FakeRunner(), {1: ["low"], 3: ["high"]}, ["benign"], [-1])
    torch.testing.assert_close(payload["vectors"][1][1], torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(payload["vectors"][3][1], torch.tensor([3.0, 4.0]))
    assert payload["counts"] == {"benign": 1, "levels": {1: 1, 3: 1}}
    path = tmp_path / "vectors.pt"
    save_vectors(payload, path)
    loaded = load_vectors(
        path,
        expected_model={
            "hf_id": FakeRunner.hf_id,
            "revision": FakeRunner.revision,
            "hidden_size": 2,
        },
    )
    torch.testing.assert_close(loaded["vectors"][3][1], payload["vectors"][3][1])
    with pytest.raises(ValueError, match="does not match"):
        load_vectors(path, expected_model={"revision": "different-model-revision"})
    normalized = compute_vectors(FakeRunner(), {3: ["high"]}, ["benign"], [0], normalize=True)
    torch.testing.assert_close(normalized["vectors"][3][0], torch.tensor([0.6, 0.8]))


def test_official_prompt_selects_context_without_chat_wrapping():
    row = {
        "behavior": "test behavior",
        "functional_category": "contextual",
        "context": "test context",
    }
    prompt = format_judge_prompt(row, "test generation")
    assert prompt == LLAMA2_CLS_PROMPT["prompt_contextual"].format(
        behavior="test behavior", context="test context", generation="test generation"
    )
    assert prompt.startswith("[INST] <<SYS>>")
    assert prompt.endswith("Answer: [/INST]")
    with pytest.raises(ValueError, match="nonempty context"):
        format_judge_prompt({**row, "context": ""}, "test generation")
    standard = format_judge_prompt(
        {"Behavior": "test behavior", "FunctionalCategory": "standard", "Tags": ""}, ""
    )
    assert "[CONTEXT]" not in standard


def test_classifier_label_errors_are_distinct_from_negatives():
    assert parse_judge_label("Yes") is True
    assert parse_judge_label(" no\n") is False
    for invalid in ("", "yes.", "not sure", "yes because", "no yes"):
        with pytest.raises(JudgeOutputError):
            parse_judge_label(invalid)


def test_prompt_copy_matches_pinned_upstream_if_source_available():
    upstream = Path(__file__).parents[1] / "work/harmbench/eval_utils.py"
    if not upstream.exists():
        pytest.skip("Upstream source is a build-time reference, not a runtime dependency.")
    nodes = ast.parse(upstream.read_text()).body
    assignment = next(
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LLAMA2_CLS_PROMPT"
            for target in node.targets
        )
    )
    assert LLAMA2_CLS_PROMPT == ast.literal_eval(assignment.value)


def test_model_placeholders_and_layer_indices_fail_before_loading():
    with pytest.raises(ValueError, match="hf_id"):
        HFRunner({"hf_id": "", "revision": "main"}, {})
    with pytest.raises(ValueError, match="revision"):
        HFRunner({"hf_id": "example/model", "revision": ""}, {})
    assert resolve_layer_index(-1, 4) == 3
    with pytest.raises(ValueError, match="outside"):
        resolve_layer_index(-5, 4)


def test_batched_extraction_matches_unpadded_single_means_and_short_final_batch(runner, torch):
    prompts = ["hello", "hello safe test", "safe test"] * 3
    runner.batch_size = 1
    singles = runner.activation_means(prompts, [-1])
    runner.batch_size = 4
    batched = runner.activation_means(prompts, [-1])
    torch.testing.assert_close(batched[1], singles[1], atol=1e-6, rtol=1e-5)
    assert all(not block._forward_hooks for block in runner.layers)


def test_batched_generation_removes_full_padded_prompt_and_preserves_order(runner, torch, monkeypatch):
    runner.batch_size = 32

    def generate(**kwargs):
        ids = kwargs["input_ids"]
        mask = kwargs["attention_mask"]
        assert ids.shape[0] == 2
        assert mask[0, 0].item() == 0
        tail = torch.tensor([[7, 2], [8, 2]], device=ids.device)
        return torch.cat([ids, tail], dim=1)

    monkeypatch.setattr(runner.model, "generate", generate)
    steering = {"layer": -1, "vector": torch.ones(8), "coefficient": 0.2}
    assert runner.generate_batch(["hello", "hello safe test"], steering) == ["safe", "unsafe"]
    assert all(not block._forward_hooks for block in runner.layers)
    with pytest.raises(ValueError, match="no truncation"):
        runner.generate_batch(["hello", "hello " * 40])


def test_batched_extraction_removes_hooks_on_forward_error(runner, monkeypatch):
    runner.batch_size = 32

    def fail(**kwargs):
        raise RuntimeError("intentional batch failure")

    monkeypatch.setattr(runner.model, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional batch failure"):
        runner.activation_means(["hello", "hello safe"], [-1])
    assert all(not block._forward_hooks for block in runner.layers)
