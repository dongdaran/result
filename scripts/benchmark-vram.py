"""Measure batch VRAM on fixed benign synthetic inputs; never change experiment runs."""

import argparse
import gc
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import transformers

from hb_steering.config import load_config, require_model, safe_name
from hb_steering.models import HFRunner, block_hidden, last_token_positions
from hb_steering.storage import atomic_json, run_lock

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model_1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--input-targets", type=int, nargs="+", default=[512, 2048])
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes + args.input_targets):
        parser.error("batch sizes and input lengths must be positive")
    safe_name(args.run_id)
    config, _ = load_config(ROOT)
    model = require_model(config, args.model)
    output = ROOT / "outputs" / args.run_id / args.model / "diagnostics"
    spec = {
        "model": model,
        "runtime": config["runtime"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "batch_sizes": args.batch_sizes,
        "input_targets": args.input_targets,
        "generation_tokens": config["runtime"]["max_new_tokens"],
        "extraction_repeats": 3,
        "generation_repeats": 1,
        "synthetic_input": "Explain a peaceful landscape. " + "blue sky " * 8,
        "generation_forces_length": True,
        "input_length_selection": "binary_search_chat_token_count_v1",
    }
    with run_lock(output):
        path = output / "vram.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["spec"] != spec:
                raise ValueError("Benchmark settings changed; use a new run ID")
            if report.get("complete"):
                print("Benchmark already complete:", path)
                return
        else:
            report = {"spec": spec, "rows": [], "complete": False}
            atomic_json(path, report)
        runner = None
        try:
            runner = HFRunner(model, config["runtime"])
            if runner.device.type != "cuda":
                raise ValueError("VRAM benchmarking requires CUDA")
            if runner.revision != model["revision"]:
                raise ValueError("Resolved model revision differs from requested commit")
            torch.cuda.synchronize()
            report["environment"] = {
                "gpu": torch.cuda.get_device_name(runner.device),
                "dtype": runner.dtype_name,
                "attention_implementation": runner.model.config._attn_implementation,
                "model_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "total_vram_gib": torch.cuda.get_device_properties(runner.device).total_memory / 2**30,
            }
            atomic_json(path, report)
            print("MODEL READY", json.dumps(report["environment"]), flush=True)
            # Warm CUDA kernels before resetting measurement counters.
            with torch.inference_mode():
                warm = runner.model(**runner._encode_prompt("Describe a blue sky."), use_cache=False)
                del warm
            for target in spec["input_targets"]:
                # Search chat-tokenized length rather than assuming tokens per word.
                low, high = 1, target
                best = None
                while low <= high:
                    repeats = (low + high) // 2
                    text = "Explain a peaceful landscape. " + "blue sky " * repeats
                    encoded = runner._encode_prompt(text)
                    length = encoded["input_ids"].shape[1]
                    candidate = (abs(length - target), repeats)
                    if best is None or candidate < best:
                        best = candidate
                    del encoded
                    if length < target:
                        low = repeats + 1
                    elif length > target:
                        high = repeats - 1
                    else:
                        break
                repeats = best[1]
                prompt = "Explain a peaceful landscape. " + "blue sky " * repeats
                single = runner._encode_prompt(prompt)
                actual = single["input_ids"].shape[1]
                for mode in ("extract_forward", "generate"):
                    for batch in spec["batch_sizes"]:
                        key = f"{mode}:{target}:{batch}"
                        if any(row["key"] == key and row["status"] == "ok" for row in report["rows"]):
                            continue
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        inputs = {k: v.repeat(batch, 1) for k, v in single.items()}
                        row = {"key": key, "mode": mode, "batch": batch, "input_tokens": actual,
                               "target_input_tokens": target, "prompt_repeats": repeats,
                               "baseline_allocated_gib": base / 2**30}
                        started = time.perf_counter()
                        try:
                            if mode == "extract_forward":
                                positions = last_token_positions(inputs["attention_mask"])
                                captured = {}

                                def capture(_module, _args, value):
                                    hidden = block_hidden(value)
                                    activation = hidden[torch.arange(batch, device=hidden.device), positions]
                                    activation = activation.detach().to(device="cpu", dtype=torch.float32)
                                    if activation.shape != (batch, runner.hidden_size) or not torch.isfinite(activation).all():
                                        raise ValueError("Invalid batch activation")
                                    captured["activation"] = activation

                                with ExitStack() as stack:
                                    handle = runner.layers[-1].register_forward_hook(capture)
                                    stack.callback(handle.remove)
                                    with torch.inference_mode():
                                        for _ in range(spec["extraction_repeats"]):
                                            result = runner.model(**inputs, use_cache=False)
                                            del result
                                    if "activation" not in captured:
                                        raise RuntimeError("Activation hook did not run")
                                del captured
                                row["new_tokens"] = 0
                            else:
                                new = spec["generation_tokens"]
                                if actual + new > runner._context_window:
                                    raise ValueError("Benchmark exceeds model context window")
                                with torch.inference_mode():
                                    result = runner.model.generate(
                                        **inputs, do_sample=False, use_cache=True,
                                        max_new_tokens=new, min_new_tokens=new,
                                        pad_token_id=runner.tokenizer.pad_token_id or runner.tokenizer.eos_token_id,
                                    )
                                if result.shape != (batch, actual + new):
                                    raise ValueError("Generation did not produce fixed token count")
                                del result
                                row["new_tokens"] = new
                            torch.cuda.synchronize()
                            row["status"] = "ok"
                        except torch.cuda.OutOfMemoryError as exc:
                            row["status"] = "oom"
                            row["error"] = str(exc)
                        except Exception as exc:
                            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                            report["rows"].append(row)
                            atomic_json(path, report)
                            raise
                        row.update(
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                            seconds=time.perf_counter() - started,
                        )
                        row["incremental_allocated_gib"] = row["peak_allocated_gib"] - row["baseline_allocated_gib"]
                        report["rows"].append(row)
                        atomic_json(path, report)
                        print("MEASURED", json.dumps(row), flush=True)
                        del inputs
                        if row["status"] != "ok":
                            raise RuntimeError("Benchmark stopped on OOM; no automatic downsizing")
                del single
            report["complete"] = True
            atomic_json(path, report)
            print("RESULT", path, flush=True)
        finally:
            if runner is not None:
                runner.close()


if __name__ == "__main__":
    main()
