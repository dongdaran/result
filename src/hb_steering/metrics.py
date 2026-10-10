"""Pooled behavior-level ASR counts and descriptive Wilson intervals.

The intervals use an independent Bernoulli model. They do not account for
correlated contexts, related behaviors, or dependence between fitted folds.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Return a two-sided Wilson score interval, or None for no observations."""
    if type(successes) is not int or type(n) is not int:
        raise ValueError("Wilson counts must be integers")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= n")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("Confidence must be finite and strictly between 0 and 1")
    if n == 0:
        return None
    # Use the lower tail to avoid rounding (1 + confidence) / 2 to exactly 1.
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    z2 = z * z
    proportion = successes / n
    denominator = 1 + z2 / n
    center = (proportion + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z2 / (4 * n * n))
    margin /= denominator
    return (
        0.0 if successes == 0 else max(0.0, center - margin),
        1.0 if successes == n else min(1.0, center + margin),
    )


def summarize_cells(
    jobs: list[dict],
    generations: dict,
    scores: dict,
    layers: list[int],
    group_ids: dict[str, str],
    *,
    evaluation_method: str = "harmbench",
) -> list[dict]:
    """Pool distinct held-out behaviors, never average per-fold percentages.

    Callers additionally validate artifact provenance. Missing jobs remain in
    the denominator plan, but ASR and its interval are withheld until the cell
    has all planned generations and boolean scores. A behavior can occur only
    once per cell, even when job keys include different fold identifiers.
    """
    if evaluation_method not in {"harmbench", "rule"}:
        raise ValueError("Unknown evaluation method")
    if not isinstance(layers, list) or any(type(layer) is not int for layer in layers):
        raise ValueError("Layers must be a list of integers")
    if len(set(layers)) != len(layers):
        raise ValueError("Layers must be unique")
    if not isinstance(jobs, list) or not isinstance(group_ids, dict):
        raise ValueError("Jobs must be a list and group IDs must be a mapping")
    if not isinstance(generations, dict) or not isinstance(scores, dict):
        raise ValueError("Generations and scores must be mappings keyed by planned jobs")

    pairs = [(0, None)] + [(source, layer) for source in (1, 2, 3) for layer in layers]
    cells = {(source, layer, target): [] for source, layer in pairs for target in (1, 2, 3)}
    seen_ids = {cell: set() for cell in cells}
    planned = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each planned job must be a mapping")
        key, ident = job.get("key"), job.get("behavior_id")
        source, layer, target = (
            job.get("source_level"),
            job.get("layer"),
            job.get("target_level"),
        )
        if not isinstance(key, str) or not key or not isinstance(ident, str) or not ident:
            raise ValueError("Job key and behavior ID must be nonempty strings")
        if key in planned:
            raise ValueError(f"Duplicate planned job key: {key}")
        if (
            type(source) is not int
            or type(target) is not int
            or (layer is not None and type(layer) is not int)
            or (source, layer, target) not in cells
        ):
            raise ValueError(f"Invalid source level, target level, or layer: {key}")
        group = group_ids.get(ident)
        if not isinstance(group, str) or not group:
            raise ValueError(f"Missing or invalid group ID for behavior: {ident}")
        cell = (source, layer, target)
        if ident in seen_ids[cell]:
            raise ValueError(f"Duplicate behavior ID within an ASR cell: {ident}")
        seen_ids[cell].add(ident)
        cells[cell].append(job)
        planned[key] = job

    if set(generations) - set(planned) or set(scores) - set(generations):
        raise ValueError("Unexpected generation/judgment keys")
    if any(not isinstance(row, dict) for row in generations.values()):
        raise ValueError("Each generation must be a mapping")
    if any(
        not isinstance(row, dict) or type(row.get("success")) is not bool for row in scores.values()
    ):
        raise ValueError("Each judgment must contain a boolean success label")
    if evaluation_method == "rule":
        _validate_rule_observations(generations, scores)

    rows = []
    for (source, layer, target), cell_jobs in cells.items():
        n_expected = len(cell_jobs)
        n_generated = sum(job["key"] in generations for job in cell_jobs)
        labels = [scores[job["key"]]["success"] for job in cell_jobs if job["key"] in scores]
        n_success = sum(labels)
        complete = n_expected > 0 and n_generated == n_expected and len(labels) == n_expected
        interval = wilson_interval(n_success, n_expected) if complete else None
        rows.append(
            {
                "source_level": source,
                "layer": layer,
                "target_level": target,
                "n_expected": n_expected,
                "n_generated": n_generated,
                "n_scored": len(labels),
                "n_success": n_success,
                "n_groups": len({group_ids[job["behavior_id"]] for job in cell_jobs}),
                "complete": complete,
                "evaluation_method": evaluation_method,
                "metric": "asr_rule" if evaluation_method == "rule" else "asr_harmbench",
                "asr": n_success / n_expected if complete else None,
                "asr_ci_low": interval[0] if interval is not None else None,
                "asr_ci_high": interval[1] if interval is not None else None,
            }
        )
    baselines = {row["target_level"]: row["asr"] for row in rows if row["source_level"] == 0}
    for row in rows:
        baseline = baselines[row["target_level"]]
        row["baseline_asr"] = baseline
        row["delta_asr"] = (
            row["asr"] - baseline if row["asr"] is not None and baseline is not None else None
        )
        for field in ("asr", "asr_ci_low", "asr_ci_high", "baseline_asr", "delta_asr"):
            explicit = field.replace("asr", "asr_rule")
            row[explicit] = row[field] if evaluation_method == "rule" else None
    return rows


def _validate_rule_observations(generations: dict, scores: dict) -> None:
    """Reject missing outputs and inconsistent refusal labels rather than count errors."""
    for generation in generations.values():
        text = generation.get("generation")
        if not isinstance(text, str) or not text.strip() or "error" in generation:
            raise ValueError("Rule metrics require successful nonempty generation strings")
    for score in scores.values():
        if "refused" in score and (
            type(score["refused"]) is not bool or score["refused"] == score["success"]
        ):
            raise ValueError("Rule refusal and success labels must be complementary booleans")


def summarize_over_refusal(
    jobs: list[dict],
    generations: dict,
    scores: dict,
    layers: list[int],
    group_ids: dict[str, str] | None = None,
    *,
    evaluation_method: str = "rule",
) -> list[dict]:
    """Count safe-prompt refusals by dataset and vector condition, with no harm levels.

    Jobs use target_level=0 solely as the safe marker. Each cell counts distinct
    prompt IDs once, and all expected generations and judgments must exist
    before a refusal rate is reported. A classifier's harmful-success label
    cannot be complemented to measure over-refusal, so only rule scores apply.
    """
    if evaluation_method != "rule":
        raise ValueError("Over-refusal requires rule refusal scores")
    if not isinstance(layers, list) or any(type(layer) is not int for layer in layers):
        raise ValueError("Layers must be a list of integers")
    if len(set(layers)) != len(layers):
        raise ValueError("Layers must be unique")
    if not isinstance(jobs, list) or not isinstance(generations, dict) or not isinstance(scores, dict):
        raise ValueError("Jobs and observations must be a list and mappings")
    if group_ids is not None and not isinstance(group_ids, dict):
        raise ValueError("Group IDs must be a mapping")
    pairs = [(0, None)] + [(source, layer) for source in (1, 2, 3) for layer in layers]
    planned, datasets, seen = {}, set(), {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each planned job must be a mapping")
        key, ident, dataset = job.get("key"), job.get("behavior_id"), job.get("dataset")
        if any(not isinstance(value, str) or not value for value in (key, ident, dataset)):
            raise ValueError("Safe job key, prompt ID, and dataset must be nonempty strings")
        if key in planned:
            raise ValueError(f"Duplicate planned job key: {key}")
        source, layer, target = job.get("source_level"), job.get("layer"), job.get("target_level")
        if (
            type(source) is not int
            or type(target) is not int
            or target != 0
            or (layer is not None and type(layer) is not int)
            or (source, layer) not in pairs
        ):
            raise ValueError("Over-refusal jobs require safe target_level=0 and valid vector conditions")
        cell = (dataset, source, layer)
        ids = seen.setdefault(cell, set())
        if ident in ids:
            raise ValueError(f"Duplicate safe prompt ID within a refusal cell: {ident}")
        ids.add(ident)
        if group_ids is not None:
            group = group_ids.get(ident)
            if not isinstance(group, str) or not group:
                raise ValueError(f"Missing or invalid group ID for safe prompt: {ident}")
        datasets.add(dataset)
        planned[key] = job
    if set(generations) - set(planned) or set(scores) - set(generations):
        raise ValueError("Unexpected generation/judgment keys")
    if any(not isinstance(row, dict) for row in generations.values()):
        raise ValueError("Each generation must be a mapping")
    if any(
        not isinstance(row, dict) or type(row.get("success")) is not bool for row in scores.values()
    ):
        raise ValueError("Each judgment must contain a boolean success label")
    _validate_rule_observations(generations, scores)
    cells = {(dataset, source, layer): [] for dataset in sorted(datasets) for source, layer in pairs}
    for job in jobs:
        cells[(job["dataset"], job["source_level"], job["layer"])].append(job)
    rows = []
    for (dataset, source, layer), cell_jobs in cells.items():
        n_expected = len(cell_jobs)
        n_generated = sum(job["key"] in generations for job in cell_jobs)
        labels = [scores[job["key"]]["success"] for job in cell_jobs if job["key"] in scores]
        n_non_refused = sum(labels)
        n_refused = len(labels) - n_non_refused
        complete = n_expected > 0 and n_generated == n_expected and len(labels) == n_expected
        rate = n_refused / n_expected if complete else None
        interval = wilson_interval(n_refused, n_expected) if complete else None
        rows.append(
            {
                "dataset": dataset,
                "source_level": source,
                "layer": layer,
                "evaluation_method": "rule",
                "metric": "over_refusal_rule",
                "n_expected": n_expected,
                "n_generated": n_generated,
                "n_scored": len(labels),
                "n_refused": n_refused,
                "n_non_refused": n_non_refused,
                "n_success": n_non_refused,
                "n_groups": len(
                    {
                        group_ids[job["behavior_id"]] if group_ids is not None else job["behavior_id"]
                        for job in cell_jobs
                    }
                ),
                "complete": complete,
                "non_refusal_rate_rule": n_non_refused / n_expected if complete else None,
                "asr_rule": n_non_refused / n_expected if complete else None,
                "asr_rule_ci_low": 1 - interval[1] if interval is not None else None,
                "asr_rule_ci_high": 1 - interval[0] if interval is not None else None,
                "over_refusal": rate,
                "over_refusal_rule": rate,
                "over_refusal_ci_low": interval[0] if interval is not None else None,
                "over_refusal_ci_high": interval[1] if interval is not None else None,
            }
        )
    baselines = {row["dataset"]: row["over_refusal"] for row in rows if row["source_level"] == 0}
    for row in rows:
        baseline = baselines[row["dataset"]]
        row["baseline_over_refusal"] = baseline
        row["baseline_asr_rule"] = 1 - baseline if baseline is not None else None
        row["delta_over_refusal"] = (
            row["over_refusal"] - baseline
            if row["over_refusal"] is not None and baseline is not None
            else None
        )
        row["delta_asr_rule"] = (
            -row["delta_over_refusal"] if row["delta_over_refusal"] is not None else None
        )
    return rows
