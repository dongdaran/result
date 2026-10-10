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


def plan_jobs(
    records: list[dict], labels: dict, layers: list[int], fold: int | None = None
) -> list[dict]:
    jobs = []
    for record in records:
        ident = record["behavior_id"]
        base = {"behavior_id": ident, "target_level": labels[ident]}
        prefix = "" if fold is None else f"fold:{fold}:"
        if fold is not None:
            base["fold"] = fold
        jobs.append({**base, "key": f"{prefix}baseline:{ident}", "source_level": 0, "layer": None})
        for source in (1, 2, 3):
            for layer in layers:
                jobs.append(
                    {
                        **base,
                        "key": f"{prefix}steered:{source}:{layer}:{ident}",
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
    fold: int | None,
    limit: int | None,
) -> dict:
    safe_name(model_key)
    safe_name(run_id)
    require_model(config, model_key)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Use a positive limit-per-level")
    records, labels, benign, saved = read_prepared(root, config)
    fold_ids = [entry["fold"] for entry in saved["harmbench"]["folds"]]
    if fold is not None and (type(fold) is not int or fold not in fold_ids):
        raise ValueError(f"Select --fold from {fold_ids}")
    benign_ids = set(saved["benign"]["fit"])
    benign_fit = sorted(
        (r for r in benign if r["benign_id"] in benign_ids), key=lambda r: r["benign_id"]
    )
    # Preserve the full benign fit set for leakage checks even during a smoke run.
    external_harmful, external_benign = [], []
    external_identity = None
    if config.get("external", {}).get("enabled", False):
        from .external import read_external

        external_harmful, external_benign = read_external(root, config, records, benign_fit)
        external_identity = {
            "manifest_sha256": file_digest(root / "data/processed/external/manifest.json"),
            "labels_sha256": file_digest(root / config["external"]["labels_path"]),
        }
    if limit:
        benign_fit = benign_fit[: 3 * limit]
    code = {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted((root / "src/hb_steering").glob("*.py"))
    }
    run_spec = {
        "schema_version": 2,
        "config": config,
        "dataset_fingerprint": saved["fingerprint"],
        "splits_sha256": file_digest(root / config["dataset"]["splits_path"]),
        "code_sha256": fingerprint(code),
        "evaluation_protocol": "grouped_kfold_out_of_fold",
        "n_folds": len(fold_ids),
        "dependency_lock_sha256": file_digest(root / "uv.lock")
        if (root / "uv.lock").exists()
        else None,
        "limit_per_level": limit,
        "external_identity": external_identity,
    }
    if config["judge"].get("method") == "rule":
        from .evaluate import rule_metadata

        run_spec["evaluator"] = rule_metadata()
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
    model_fingerprint = fingerprint({**run_spec, "model_key": model_key})
    common = {
        "root": root,
        "directory": directory / model_key,
        "config": config,
        "model_key": model_key,
        "records": records,
        "labels": labels,
        "benign_fit": benign_fit,
        "group_ids": saved["harmbench"]["group_ids"],
        "limit_per_level": limit,
        "fingerprint": model_fingerprint,
    }
    fold_runs = []
    for entry in saved["harmbench"]["folds"]:
        index = entry["fold"]
        evaluation = selected_records(records, labels, entry["test"], limit)
        fold_runs.append(
            {
                **common,
                "directory": common["directory"] / f"fold_{index}",
                "fold": index,
                "fit": selected_records(records, labels, entry["fit"], limit),
                "evaluation": evaluation,
                "jobs": plan_jobs(evaluation, labels, config["steering"]["layers"], index),
                "fingerprint": fingerprint({"model_fingerprint": model_fingerprint, "fold": index}),
            }
        )
    result = {
        **common,
        "all_fold_runs": fold_runs,
        "fold_runs": [run for run in fold_runs if fold is None or run["fold"] == fold],
        "selected_fold": fold,
    }
    if config.get("external", {}).get("enabled", False) or config.get("evaluation", {}).get(
        "over_refusal", False
    ):
        result["transfer_run"] = _setup_transfer(
            common, records, labels, benign, saved, external_harmful, external_benign, limit
        )
    return result


def _setup_transfer(
    common: dict,
    records: list[dict],
    labels: dict,
    benign: list[dict],
    saved: dict,
    external_harmful: list[dict],
    external_benign: list[dict],
    limit: int | None,
) -> dict:
    """One final HB-only vector set, evaluated on independent external/safe prompts."""
    harmful_labels = {r["behavior_id"]: r["level"] for r in external_harmful}
    evaluation = (
        selected_records(external_harmful, harmful_labels, list(harmful_labels), limit)
        if external_harmful
        else []
    )
    if common["config"].get("evaluation", {}).get("over_refusal", False):
        test_ids = set(saved["benign"]["test"])
        safe = [
            {
                "behavior_id": f"xstest_safe:{r['benign_id']}",
                "behavior": r["prompt"],
                "context": "",
                "functional_category": "standard",
                "level": 0,
                "dataset": "xstest_safe",
                "group_id": f"xstest_safe:{r['benign_id']}",
            }
            for r in benign
            if r["benign_id"] in test_ids
        ]
        for dataset_rows in (safe, external_benign):
            ordered = sorted(dataset_rows, key=lambda r: r["behavior_id"])
            evaluation.extend(ordered[: 3 * limit] if limit else ordered)
    if not evaluation:
        raise ValueError("Transfer evaluation has no independent prompts")
    eval_labels = {r["behavior_id"]: r["level"] for r in evaluation}
    if len(eval_labels) != len(evaluation) or set(eval_labels) & set(labels):
        raise ValueError("Transfer evaluation IDs overlap each other or HarmBench fit IDs")
    jobs = plan_jobs(evaluation, eval_labels, common["config"]["steering"]["layers"])
    dataset_by_id = {r["behavior_id"]: r["dataset"] for r in evaluation}
    for job in jobs:
        job["dataset"] = dataset_by_id[job["behavior_id"]]
        job["key"] = "transfer:" + job["key"]
    return {
        **common,
        "directory": common["directory"] / "transfer",
        "records": records + evaluation,
        "labels": {**labels, **eval_labels},
        "fit": selected_records(records, labels, list(labels), limit),
        "evaluation": evaluation,
        "jobs": jobs,
        "group_ids": {
            r["behavior_id"]: r.get("group_id", r["behavior_id"]) for r in evaluation
        },
        "fingerprint": fingerprint(
            {"model_fingerprint": common["fingerprint"], "scope": "final_external_transfer"}
        ),
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
    if "fold" in run and payload.get("fold") != run["fold"]:
        raise ValueError("Vector artifact fold identity differs from the requested fold")
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
        if "fold" in run:
            payload["fold"] = run["fold"]
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
    ) or not row["generation"].strip():
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
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Generation did not return nonempty text")
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
    from .evaluate import HarmBenchJudge, RuleJudge

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
        judge_config = require_judge(run["config"])
        judge = (
            RuleJudge(judge_config)
            if judge_config.get("method") == "rule"
            else HarmBenchJudge(judge_config)
        )
    failed = 0
    try:
        if owned and hasattr(judge, "runner"):
            _environment(judge.runner, run["directory"] / "logs/judge_environment.json")
        elif hasattr(judge, "metadata"):
            atomic_json(run["directory"] / "logs/rule_evaluator.json", judge.metadata)
        behavior_map = {r["behavior_id"]: r for r in run["records"]}
        for index, key in enumerate(pending, 1):
            generation = generations[key]
            try:
                details = (
                    judge.score(generation["generation"])
                    if isinstance(judge, RuleJudge)
                    else {}
                )
                success = details.get("success") if details else judge.judge(
                    behavior_map[generation["behavior_id"]], generation["generation"]
                )
                if type(success) is not bool:
                    raise ValueError("Judge must return a boolean label")
                row = {
                    **planned[key],
                    "run_fingerprint": run["fingerprint"],
                    "generation_sha256": generation["generation_sha256"],
                    "success": success,
                    **details,
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
    fields = (
        "key",
        "fold",
        "behavior_id",
        "target_level",
        "source_level",
        "layer",
        "generation_sha256",
        "dataset",
    )
    if (
        any(row.get(k) != generation.get(k) for k in fields)
        or row.get("run_fingerprint") != run["fingerprint"]
        or type(row.get("success")) is not bool
    ):
        raise ValueError("Judgment provenance/schema mismatch")
    if run["config"]["judge"].get("method") == "rule":
        from .evaluate import RuleJudge

        expected = RuleJudge(run["config"]["judge"]).score(generation["generation"])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Rule judgment differs from the frozen evaluator or response")


def _validated_records(run: dict) -> tuple[dict, dict]:
    generations = read_journal(run["directory"] / "generations/responses.jsonl")
    scores = read_journal(run["directory"] / "metrics/judgments.jsonl")
    planned = {job["key"]: job for job in run["jobs"]}
    if set(generations) - set(planned) or set(scores) - set(generations):
        raise ValueError("Unexpected generation/judgment keys")
    for key, row in generations.items():
        _validate_generation(row, planned[key], run)
    for key, row in scores.items():
        _validate_score(row, generations[key], run)
    return generations, scores


def _write_report(directory: Path, report: dict, layers: list[int]) -> None:
    directory = directory / "metrics"
    rows = report["cells"]
    atomic_json(directory / "summary.json", report)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_data = stream.getvalue().encode()
    atomic_bytes(directory / "asr.csv", csv_data)
    metric = "ASR_rule" if report.get("evaluation_method") == "rule" else "ASR_HarmBench"
    if metric == "ASR_rule":
        atomic_bytes(directory / "asr_rule.csv", csv_data)

    def display(cell):
        if cell["asr"] is None:
            return f"incomplete ({cell['n_scored']}/{cell['n_expected']} scored)"
        return (
            f"{cell['asr'] * 100:.2f}% ({cell['n_success']}/{cell['n_expected']}; "
            f"95% CI {cell['asr_ci_low'] * 100:.2f}–{cell['asr_ci_high'] * 100:.2f}%)"
        )

    md = [
        f"# {metric}",
        "",
        f"Scope: {report['evaluation_scope']}; complete={report['complete']}; "
        f"full_oof={report.get('full_oof', False)}.",
        "",
        f"Scored {report['scored_jobs']}/{report['expected_jobs']} planned suite jobs.",
        "",
        "Rows are source-vector levels; columns are evaluation prompt levels.",
        f"{metric} pools success counts and denominators, never unweighted fold means.",
        "",
        "95% Wilson intervals are binomial diagnostics assuming independent behaviors. "
        "They do not capture shared-context dependence or uncertainty from overlapping "
        "training folds and vector estimation. n_groups is available in JSON/CSV.",
        "",
    ]
    if metric == "ASR_rule":
        md.extend([
            "ASR_rule is the fraction without a matched refusal phrase. It is a non-refusal "
            "proxy and does not establish harmful compliance or answer quality.",
            "",
        ])
    for cell in rows:
        if cell["source_level"] == 0:
            md.append(f"Baseline L{cell['target_level']}: {display(cell)}")
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
            cells = [row for row in rows if row["source_level"] == source and row["layer"] == layer]
            md.append(f"| L{source} | " + " | ".join(display(cell) for cell in cells) + " |")
    md_data = ("\n".join(md) + "\n").encode()
    atomic_bytes(directory / "asr.md", md_data)
    if metric == "ASR_rule":
        atomic_bytes(directory / "asr_rule.md", md_data)


def _make_report(
    run: dict,
    jobs: list[dict],
    generations: dict,
    scores: dict,
    scope: str,
    *,
    full_oof: bool = False,
) -> dict:
    from .metrics import summarize_cells

    group_ids = run.get("group_ids", {job["behavior_id"]: job["behavior_id"] for job in jobs})
    method = "rule" if run["config"]["judge"].get("method") == "rule" else "harmbench"
    rows = summarize_cells(
        jobs, generations, scores, run["config"]["steering"]["layers"], group_ids,
        evaluation_method=method,
    )
    complete = len(scores) == len(jobs) and all(row["complete"] for row in rows)
    return {
        "schema_version": 2,
        "evaluation_method": method,
        "metric": "asr_rule" if method == "rule" else "asr_harmbench",
        "complete": complete,
        "full_oof": full_oof and complete,
        "evaluation_scope": scope,
        "run_fingerprint": run["fingerprint"],
        "asr_units": "fraction_0_to_1",
        "delta_units": "fraction_points",
        "matrix_rows": "source_level",
        "matrix_columns": "target_level",
        "ci_method": "wilson_95_iid_binomial_diagnostic",
        "ci_limitation": "Does not account for shared contexts, overlapping fit folds, or vector-estimation uncertainty",
        "expected_jobs": len(jobs),
        "generated_jobs": len(generations),
        "scored_jobs": len(scores),
        "cells": rows,
    }


def summarize(run: dict) -> dict:
    generations, scores = _validated_records(run)
    report = _make_report(run, run["jobs"], generations, scores, "single_fold")
    report["fold"] = run.get("fold")
    _write_report(run["directory"], report, run["config"]["steering"]["layers"])
    return report


def summarize_oof(run: dict) -> dict:
    jobs, generations, scores = [], {}, {}
    completed = []
    for fold in run["all_fold_runs"]:
        fold_generations, fold_scores = _validated_records(fold)
        jobs.extend(fold["jobs"])
        if set(generations) & set(fold_generations) or set(scores) & set(fold_scores):
            raise ValueError("Duplicate journal keys across folds")
        generations.update(fold_generations)
        scores.update(fold_scores)
        if len(fold_scores) == len(fold["jobs"]):
            completed.append(fold["fold"])
    report = _make_report(
        run, jobs, generations, scores, "pooled_oof", full_oof=run["limit_per_level"] is None
    )
    report.update(
        folds_expected=[fold["fold"] for fold in run["all_fold_runs"]],
        folds_completed=completed,
        limit_per_level=run["limit_per_level"],
        interpretation="Performance of the fold-specific vector-building procedure, not one final fixed vector",
    )
    _write_report(run["directory"], report, run["config"]["steering"]["layers"])
    return report


def summarize_transfer(run: dict) -> dict:
    """Report external harmful ASR_rule and independent safe over-refusal separately."""
    from .metrics import summarize_over_refusal

    generations, scores = _validated_records(run)
    harmful_jobs = [job for job in run["jobs"] if job["target_level"] in (1, 2, 3)]
    safe_jobs = [job for job in run["jobs"] if job["target_level"] == 0]

    def subset(rows, jobs):
        keys = {job["key"] for job in jobs}
        return {key: row for key, row in rows.items() if key in keys}

    layers = run["config"]["steering"]["layers"]
    report = (
        _make_report(
            run, harmful_jobs, subset(generations, harmful_jobs), subset(scores, harmful_jobs),
            "final_external_transfer",
        )
        if harmful_jobs
        else {"cells": [], "evaluation_method": "rule"}
    )
    safe_cells = (
        summarize_over_refusal(
            safe_jobs, subset(generations, safe_jobs), subset(scores, safe_jobs), layers,
            run["group_ids"], evaluation_method="rule",
        )
        if safe_jobs
        else []
    )
    complete = len(scores) == len(run["jobs"]) and all(
        row["complete"] for row in report["cells"] + safe_cells
    )
    report.update(
        schema_version=3,
        evaluation_scope="final_external_transfer",
        run_fingerprint=run["fingerprint"],
        complete=complete,
        full_transfer=complete and run["limit_per_level"] is None,
        limit_per_level=run["limit_per_level"],
        fit_scope="all_harmbench_only",
        fit_counts={
            str(level): sum(run["labels"][r["behavior_id"]] == level for r in run["fit"])
            for level in (1, 2, 3)
        },
        harmful_datasets=sorted({job["dataset"] for job in harmful_jobs}),
        expected_jobs=len(run["jobs"]),
        generated_jobs=len(generations),
        scored_jobs=len(scores),
        harmful_expected_jobs=len(harmful_jobs),
        safe_expected_jobs=len(safe_jobs),
        over_refusal_cells=safe_cells,
        interpretation="One final HB-only vector set; external and safe prompts never fit the vectors",
    )
    if report["cells"]:
        _write_report(run["directory"], report, layers)
    directory = run["directory"] / "metrics"
    if safe_cells:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(safe_cells[0]))
        writer.writeheader()
        writer.writerows(safe_cells)
        atomic_bytes(directory / "over_refusal.csv", stream.getvalue().encode())
        md = [
            "# Rule-based over-refusal", "",
            "On safe prompts: over_refusal = 1 - ASR_rule. Refusal phrases are a heuristic.",
            "", "| Dataset | Vector | Block | ASR_rule | Over-refusal | Scored / expected |",
            "|---|---|---|---:|---:|---:|",
        ]
        for cell in safe_cells:
            asr = cell["asr_rule"]
            refusal = cell["over_refusal"]
            source = "baseline" if cell["source_level"] == 0 else f"L{cell['source_level']}"
            md.append(
                f"| {cell['dataset']} | {source} | {cell['layer']} | "
                + (f"{asr:.2%} | {refusal:.2%}" if asr is not None else "incomplete | incomplete")
                + f" | {cell['n_scored']} / {cell['n_expected']} |"
            )
        atomic_bytes(directory / "over_refusal.md", ("\n".join(md) + "\n").encode())
    atomic_json(directory / "summary.json", report)
    return report
