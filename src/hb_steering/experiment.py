from __future__ import annotations

import csv
import io
import json
import platform
import uuid
from pathlib import Path

from .config import require_judge, require_model, safe_name
from .prepare import read_prepared
from .storage import (
    append_record,
    atomic_bytes,
    atomic_json,
    file_digest,
    fingerprint,
    read_journal,
    run_lock,
)


def behavior_prompt(record: dict) -> str:
    context = record["context"].strip()
    return f"{context}\n\n---\n\n{record['behavior']}" if context else record["behavior"]


def selected_records(records: list[dict], labels: dict, ids: list[str], limit: int | None):
    selected = []
    for level in (1, 2, 3):
        rows = sorted(
            (r for r in records if r["behavior_id"] in ids and labels[r["behavior_id"]] == level),
            key=lambda r: r["behavior_id"],
        )
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"Selected split has no level {level} records")
        selected.extend(rows)
    return selected


def plan_jobs(records: list[dict], labels: dict, layers: list[int]) -> list[dict]:
    jobs = []
    for record in records:
        ident = record["behavior_id"]
        base = {"behavior_id": ident, "target_level": labels[ident]}
        jobs.append({**base, "key": f"baseline:{ident}", "source_level": 0, "layer": None})
        for source in (1, 2, 3):
            for layer in layers:
                jobs.append(
                    {
                        **base,
                        "key": f"steered:{source}:{layer}:{ident}",
                        "source_level": source,
                        "layer": layer,
                    }
                )
    return jobs


def setup_run(
    root: Path,
    config: dict,
    config_path: Path,
    model_key: str,
    run_id: str,
    split: str,
    limit: int | None,
) -> dict:
    safe_name(model_key)
    safe_name(run_id)
    require_model(config, model_key)
    if split not in {"dev", "test"} or (limit is not None and limit <= 0):
        raise ValueError("Use split dev/test and a positive limit-per-level")
    records, labels, benign, saved = read_prepared(root, config)
    fit = selected_records(records, labels, saved["harmbench"]["fit"], limit)
    evaluation = selected_records(records, labels, saved["harmbench"][split], limit)
    benign_ids = set(saved["benign"]["fit"])
    benign_fit = sorted(
        (r for r in benign if r["benign_id"] in benign_ids), key=lambda r: r["benign_id"]
    )
    if limit:
        benign_fit = benign_fit[: 3 * limit]
    code = {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted((root / "src/hb_steering").glob("*.py"))
    }
    run_spec = {
        "schema_version": 1,
        "config": config,
        "dataset_fingerprint": saved["fingerprint"],
        "splits_sha256": file_digest(root / config["dataset"]["splits_path"]),
        "code_sha256": fingerprint(code),
        "evaluation_split": split,
        "limit_per_level": limit,
    }
    directory = root / "outputs" / run_id
    with run_lock(directory):
        path = directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != run_spec:
                raise ValueError("Run settings/data/code changed; use a new --run-id")
        else:
            atomic_json(path, run_spec)
            atomic_bytes(directory / "config.toml", config_path.read_bytes())
            atomic_json(directory / "effective_config.json", config)
            atomic_json(directory / "code_hashes.json", code)
    return {
        "root": root,
        "directory": directory / model_key,
        "config": config,
        "model_key": model_key,
        "records": records,
        "labels": labels,
        "fit": fit,
        "evaluation": evaluation,
        "benign_fit": benign_fit,
        "jobs": plan_jobs(evaluation, labels, config["steering"]["layers"]),
        "fingerprint": fingerprint({**run_spec, "model_key": model_key}),
    }


def _environment(runner, path: Path):
    import torch
    import transformers

    atomic_json(
        path,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model": runner.hf_id,
            "revision": runner.revision,
            "dtype": runner.dtype_name,
            "device": str(runner.device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(runner.device)
            if runner.device.type == "cuda"
            else None,
        },
    )


def _vector_path(run: dict) -> Path:
    return run["directory"] / "vectors/directions.pt"


def _load_vectors(run: dict, runner=None) -> dict:
    from .vectors import load_vectors

    path = _vector_path(run)
    model = run["config"]["models"][run["model_key"]]
    expected = {"hf_id": model["hf_id"], "revision": model["revision"]}
    if runner:
        expected.update(revision=runner.revision, hidden_size=runner.hidden_size)
    payload = load_vectors(path, expected_model=expected)
    if payload.get("run_fingerprint") != run["fingerprint"]:
        raise ValueError("Vector artifact belongs to a different run")
    if payload.get("requested_layers") != run["config"]["steering"]["layers"] or set(
        payload["vectors"]
    ) != {1, 2, 3}:
        raise ValueError("Vector artifact levels/layers differ from the frozen run")
    if runner:
        from .steering import resolve_layer_index

        expected_layers = [
            resolve_layer_index(index, len(runner.layers))
            for index in run["config"]["steering"]["layers"]
        ]
        if payload["layers"] != expected_layers:
            raise ValueError("Resolved vector layers differ from the configured decoder blocks")
    meta_path = path.with_suffix(".json")
    metadata = {"sha256": file_digest(path), "run_fingerprint": run["fingerprint"]}
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Vector artifact checksum/run identity mismatch")
    if not meta_path.exists():
        atomic_json(meta_path, metadata)
    if payload["normalized"] != run["config"]["steering"]["normalize"]:
        raise ValueError("Vector normalization differs from the frozen run")
    return payload


def extract(run: dict, runner=None) -> dict:
    from .models import HFRunner
    from .vectors import compute_vectors, save_vectors

    path = _vector_path(run)
    if path.exists():
        return _load_vectors(run, runner)
    owned = runner is None
    if owned:
        runner = HFRunner(require_model(run["config"], run["model_key"]), run["config"]["runtime"])
    try:
        _environment(runner, run["directory"] / "logs/target_environment.json")
        prompts = {
            level: [
                behavior_prompt(r) for r in run["fit"] if run["labels"][r["behavior_id"]] == level
            ]
            for level in (1, 2, 3)
        }
        payload = compute_vectors(
            runner,
            prompts,
            [r["prompt"] for r in run["benign_fit"]],
            run["config"]["steering"]["layers"],
            normalize=run["config"]["steering"]["normalize"],
        )
        payload["run_fingerprint"] = run["fingerprint"]
        payload["requested_layers"] = run["config"]["steering"]["layers"]
        save_vectors(payload, path)
        return _load_vectors(run, runner)
    finally:
        if owned:
            runner.close()


def _log_error(run: dict, stage: str, key: str, exc: Exception):
    append_record(
        run["directory"] / "logs/errors.jsonl",
        {
            "key": uuid.uuid4().hex,
            "stage": stage,
            "job": key,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )
    print(f"{stage} failed: {key} ({type(exc).__name__}); details in logs/errors.jsonl", flush=True)


def _validate_generation(row: dict, job: dict, run: dict):
    if any(row.get(k) != v for k, v in job.items()):
        raise ValueError(f"Generation job metadata mismatch: {job['key']}")
    if row.get("run_fingerprint") != run["fingerprint"] or not isinstance(
        row.get("generation"), str
    ):
        raise ValueError("Generation provenance/schema mismatch")
    if row.get("generation_sha256") != fingerprint(row["generation"]):
        raise ValueError("Generation content checksum mismatch")


def generate(run: dict, runner=None) -> dict:
    from .models import HFRunner
    from .steering import resolve_layer_index

    path = run["directory"] / "generations/responses.jsonl"
    existing = read_journal(path)
    planned = {j["key"]: j for j in run["jobs"]}
    if set(existing) - set(planned):
        raise ValueError("Unexpected generation keys in this run")
    for key, row in existing.items():
        _validate_generation(row, planned[key], run)
    pending = [j for j in run["jobs"] if j["key"] not in existing]
    if not pending:
        print(f"Generation complete: {len(existing)}/{len(planned)} (reused)")
        return existing
    owned = runner is None
    if owned:
        runner = HFRunner(require_model(run["config"], run["model_key"]), run["config"]["runtime"])
    failed = 0
    try:
        _environment(runner, run["directory"] / "logs/target_environment.json")
        vectors = _load_vectors(run, runner)
        behavior_map = {r["behavior_id"]: r for r in run["records"]}
        for index, job in enumerate(pending, 1):
            steering = None
            if job["source_level"]:
                resolved = resolve_layer_index(job["layer"], len(runner.layers))
                steering = {
                    "layer": resolved,
                    "vector": vectors["vectors"][job["source_level"]][resolved],
                    "coefficient": run["config"]["steering"]["coefficient"],
                    "normalize": False,
                }  # Already normalized at extraction if configured.
            try:
                text = runner.generate(behavior_prompt(behavior_map[job["behavior_id"]]), steering)
                if not isinstance(text, str):
                    raise ValueError("Generation did not return text")
                row = {
                    **job,
                    "run_fingerprint": run["fingerprint"],
                    "generation": text,
                    "generation_sha256": fingerprint(text),
                }
                append_record(path, row)
                existing[job["key"]] = row
            except (ValueError, RuntimeError) as exc:
                _log_error(run, "generate", job["key"], exc)
                failed += 1
                if "out of memory" in str(exc).casefold():
                    raise RuntimeError(
                        "GPU out of memory; inspect logs and use a new run with smaller settings"
                    ) from exc
            if index % 10 == 0 or index == len(pending):
                print(f"Generation progress: {len(existing)}/{len(planned)}", flush=True)
    finally:
        if owned:
            runner.close()
    if failed:
        raise RuntimeError(
            f"{failed} generation jobs failed; rerun the same command to retry missing jobs"
        )
    return existing


def score(run: dict, judge=None) -> dict:
    from .evaluate import HarmBenchJudge

    generations = read_journal(run["directory"] / "generations/responses.jsonl")
    planned = {j["key"]: j for j in run["jobs"]}
    if set(generations) != set(planned):
        raise ValueError("Complete generation before scoring; rerun ./scripts/generate")
    for key, row in generations.items():
        _validate_generation(row, planned[key], run)
    path = run["directory"] / "metrics/judgments.jsonl"
    existing = read_journal(path)
    if set(existing) - set(planned):
        raise ValueError("Unexpected scoring keys")
    for key, row in existing.items():
        _validate_score(row, generations[key], run)
    pending = [key for key in planned if key not in existing]
    if not pending:
        print(f"Scoring complete: {len(existing)}/{len(planned)} (reused)")
        return existing
    owned = judge is None
    if owned:
        judge = HarmBenchJudge(require_judge(run["config"]))
    failed = 0
    try:
        if owned:
            _environment(judge.runner, run["directory"] / "logs/judge_environment.json")
        behavior_map = {r["behavior_id"]: r for r in run["records"]}
        for index, key in enumerate(pending, 1):
            generation = generations[key]
            try:
                success = judge.judge(
                    behavior_map[generation["behavior_id"]], generation["generation"]
                )
                if type(success) is not bool:
                    raise ValueError("Judge must return a boolean label")
                row = {
                    **planned[key],
                    "run_fingerprint": run["fingerprint"],
                    "generation_sha256": generation["generation_sha256"],
                    "success": success,
                }
                append_record(path, row)
                existing[key] = row
            except (ValueError, RuntimeError) as exc:
                _log_error(run, "score", key, exc)
                failed += 1
                if "out of memory" in str(exc).casefold():
                    raise RuntimeError("Judge GPU out of memory; inspect logs") from exc
            if index % 10 == 0 or index == len(pending):
                print(f"Scoring progress: {len(existing)}/{len(planned)}", flush=True)
    finally:
        if owned:
            judge.close()
    if failed:
        raise RuntimeError(
            f"{failed} judge jobs failed; rerun the same command to retry missing jobs"
        )
    return existing


def _validate_score(row: dict, generation: dict, run: dict):
    fields = ("key", "behavior_id", "target_level", "source_level", "layer", "generation_sha256")
    if (
        any(row.get(k) != generation.get(k) for k in fields)
        or row.get("run_fingerprint") != run["fingerprint"]
        or type(row.get("success")) is not bool
    ):
        raise ValueError("Judgment provenance/schema mismatch")


def summarize(run: dict) -> dict:
    generations = read_journal(run["directory"] / "generations/responses.jsonl")
    scores = read_journal(run["directory"] / "metrics/judgments.jsonl")
    planned = {j["key"]: j for j in run["jobs"]}
    if set(generations) - set(planned) or set(scores) - set(generations):
        raise ValueError("Unexpected generation/judgment keys")
    for key, row in generations.items():
        _validate_generation(row, planned[key], run)
    for key, row in scores.items():
        _validate_score(row, generations[key], run)
    rows = []
    layers = run["config"]["steering"]["layers"]
    for source, layer in [(0, None)] + [
        (source, block) for source in (1, 2, 3) for block in layers
    ]:
        for target in (1, 2, 3):
            jobs = [
                j
                for j in run["jobs"]
                if (j["source_level"], j["layer"], j["target_level"]) == (source, layer, target)
            ]
            labels = [scores[j["key"]]["success"] for j in jobs if j["key"] in scores]
            complete = len(labels) == len(jobs)
            rows.append(
                {
                    "source_level": source,
                    "layer": layer,
                    "target_level": target,
                    "n_expected": len(jobs),
                    "n_generated": sum(j["key"] in generations for j in jobs),
                    "n_scored": len(labels),
                    "n_success": sum(labels),
                    "complete": complete,
                    "asr": sum(labels) / len(jobs) if complete and jobs else None,
                }
            )
    baselines = {r["target_level"]: r["asr"] for r in rows if r["source_level"] == 0}
    for row in rows:
        baseline = baselines[row["target_level"]]
        row["baseline_asr"] = baseline
        row["delta_asr"] = (
            row["asr"] - baseline if baseline is not None and row["asr"] is not None else None
        )
    complete = len(scores) == len(planned)
    report = {
        "schema_version": 1,
        "complete": complete,
        "run_fingerprint": run["fingerprint"],
        "asr_units": "fraction_0_to_1",
        "delta_units": "fraction_points",
        "matrix_rows": "source_level",
        "matrix_columns": "target_level",
        "expected_jobs": len(planned),
        "generated_jobs": len(generations),
        "scored_jobs": len(scores),
        "cells": rows,
    }
    directory = run["directory"] / "metrics"
    atomic_json(directory / "summary.json", report)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(directory / "asr.csv", stream.getvalue().encode())
    md = [
        "# ASR",
        "",
        f"Complete: {complete}; scored {len(scores)}/{len(planned)}.",
        "",
        "Values are ASR percentages; rows are source levels, columns are evaluation levels.",
        "",
    ]

    def display(value):
        return f"{value * 100:.2f}%" if value is not None else "incomplete"

    md.append("Baseline: " + ", ".join(f"L{t}={display(baselines[t])}" for t in (1, 2, 3)))
    for layer in layers:
        md.extend(
            [
                "",
                f"## Decoder block {layer}",
                "",
                "| Source | L1 | L2 | L3 |",
                "|---|---:|---:|---:|",
            ]
        )
        for source in (1, 2, 3):
            cells = [r for r in rows if r["source_level"] == source and r["layer"] == layer]
            md.append(f"| L{source} | " + " | ".join(display(r["asr"]) for r in cells) + " |")
    atomic_bytes(directory / "asr.md", ("\n".join(md) + "\n").encode())
    return report
