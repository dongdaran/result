"""Behavior-level pooling and descriptive uncertainty need no GPU or network."""

from copy import deepcopy

import pytest

from hb_steering.metrics import summarize_cells, wilson_interval


def make_inputs(layers=(-1,), counts=(4, 2, 1)):
    jobs, groups = [], {}
    for target, count in enumerate(counts, 1):
        for index in range(count):
            ident = f"level{target}-behavior{index}"
            groups[ident] = f"level{target}-group{index}"
            fold = 0 if index == 0 else 1
            for source, layer in [(0, None)] + [
                (source, layer) for source in (1, 2, 3) for layer in layers
            ]:
                jobs.append(
                    {
                        "key": f"fold{fold}:{source}:{layer}:{ident}",
                        "behavior_id": ident,
                        "source_level": source,
                        "layer": layer,
                        "target_level": target,
                        "fold": fold,
                    }
                )
    generations = {job["key"]: {**job, "generation": "fixture"} for job in jobs}
    scores = {
        job["key"]: {**job, "success": job["behavior_id"].endswith(("0", "1"))} for job in jobs
    }
    return jobs, generations, scores, list(layers), groups


def indexed_rows(inputs):
    return {
        (row["source_level"], row["layer"], row["target_level"]): row
        for row in summarize_cells(*inputs)
    }


def test_wilson_known_small_sample_and_count_edges():
    assert wilson_interval(3, 6) == pytest.approx((0.18761630648, 0.81238369352))
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 6) == pytest.approx((0.0, 0.390334287902))
    assert wilson_interval(6, 6) == pytest.approx((0.609665712098, 1.0))
    assert wilson_interval(0, 1)[0] == 0.0
    assert wilson_interval(1, 1)[1] == 1.0
    lower, upper = wilson_interval(50, 100, confidence=0.99)
    low95, high95 = wilson_interval(50, 100)
    assert lower < low95 < high95 < upper


@pytest.mark.parametrize("successes,n", [(1, 0), (-1, 2), (3, 2), (0, -1), (True, 1), (0, 1.0)])
def test_wilson_rejects_invalid_counts(successes, n):
    with pytest.raises(ValueError, match="counts"):
        wilson_interval(successes, n)


@pytest.mark.parametrize("confidence", [0, 1, -0.5, 2, float("nan"), float("inf"), True, "0.95"])
def test_wilson_rejects_invalid_confidence_even_without_observations(confidence):
    with pytest.raises(ValueError, match="Confidence"):
        wilson_interval(0, 0, confidence)


def test_pooled_asr_uses_success_counts_instead_of_unweighted_fold_mean():
    inputs = make_inputs()
    jobs, _generations, scores, _layers, _groups = inputs
    row = indexed_rows(inputs)[(1, -1, 1)]
    # The uneven folds have 1/1 and 1/3 successes. Their mean is 2/3,
    # while each distinct held-out behavior gets one vote in the pooled 2/4.
    fold_rates = []
    for fold in (0, 1):
        fold_jobs = [
            job
            for job in jobs
            if job["fold"] == fold and job["source_level"] == 1 and job["target_level"] == 1
        ]
        fold_rates.append(sum(scores[job["key"]]["success"] for job in fold_jobs) / len(fold_jobs))
    assert sum(fold_rates) / len(fold_rates) == pytest.approx(2 / 3)
    assert row["n_expected"] == row["n_generated"] == row["n_scored"] == 4
    assert row["n_success"] == 2
    assert row["asr"] == row["baseline_asr"] == 0.5
    assert row["delta_asr"] == 0.0
    assert (row["asr_ci_low"], row["asr_ci_high"]) == wilson_interval(2, 4)


def test_three_baseline_rows_and_nine_transfer_rows_per_layer():
    inputs = make_inputs(layers=(-1, 0))
    rows = summarize_cells(*inputs)
    assert len(rows) == 3 + 9 * 2
    assert [row["target_level"] for row in rows[:3]] == [1, 2, 3]
    assert all(row["source_level"] == 0 and row["layer"] is None for row in rows[:3])
    assert all(row["complete"] for row in rows)
    assert {(row["source_level"], row["layer"], row["target_level"]) for row in rows[3:]} == {
        (source, layer, target) for source in (1, 2, 3) for layer in (-1, 0) for target in (1, 2, 3)
    }


def test_group_counts_are_distinct_context_groups_not_behavior_denominator():
    inputs = make_inputs()
    inputs[-1]["level1-behavior1"] = inputs[-1]["level1-behavior0"]
    for source, layer in [(0, None), (1, -1), (2, -1), (3, -1)]:
        row = indexed_rows(inputs)[(source, layer, 1)]
        assert row["n_groups"] == 3
        assert row["n_expected"] == 4
        assert row["asr"] == 0.5


def test_missing_judgment_withholds_only_its_cell_asr_and_uncertainty():
    inputs = make_inputs()
    key = next(
        job["key"] for job in inputs[0] if job["source_level"] == 1 and job["target_level"] == 1
    )
    del inputs[2][key]
    cells = indexed_rows(inputs)
    row = cells[(1, -1, 1)]
    assert row["complete"] is False
    assert row["n_expected"] == row["n_generated"] == 4
    assert row["n_scored"] == 3
    assert row["asr"] is row["asr_ci_low"] is row["asr_ci_high"] is row["delta_asr"] is None
    assert row["baseline_asr"] == 0.5
    assert cells[(2, -1, 1)]["asr"] == 0.5


def test_missing_generation_and_baseline_propagate_null_delta():
    inputs = make_inputs()
    key = next(
        job["key"] for job in inputs[0] if job["source_level"] == 0 and job["target_level"] == 1
    )
    del inputs[1][key]
    del inputs[2][key]
    cells = indexed_rows(inputs)
    baseline = cells[(0, None, 1)]
    assert baseline["n_generated"] == baseline["n_scored"] == 3
    assert baseline["complete"] is False
    assert baseline["asr"] is baseline["asr_ci_low"] is baseline["asr_ci_high"] is None
    assert cells[(1, -1, 1)]["asr"] == 0.5
    assert cells[(1, -1, 1)]["baseline_asr"] is cells[(1, -1, 1)]["delta_asr"] is None


def test_empty_cells_cannot_report_complete_or_zero_asr():
    rows = summarize_cells([], {}, {}, [-1], {})
    assert len(rows) == 12
    assert all(row["n_expected"] == row["n_groups"] == 0 for row in rows)
    assert all(row["complete"] is False and row["asr"] is None for row in rows)


def test_repeated_behavior_cannot_be_counted_again_under_another_fold():
    inputs = make_inputs()
    duplicate = deepcopy(inputs[0][0])
    duplicate.update(key="fold99:repeated", fold=99)
    inputs[0].append(duplicate)
    with pytest.raises(ValueError, match="Duplicate behavior ID"):
        summarize_cells(*inputs)


def test_duplicate_job_keys_are_rejected_before_counting():
    inputs = make_inputs()
    inputs[0].append(deepcopy(inputs[0][0]))
    with pytest.raises(ValueError, match="Duplicate planned job key"):
        summarize_cells(*inputs)


@pytest.mark.parametrize("success", [0, 1, None, "yes", [], {}])
def test_success_requires_a_literal_boolean(success):
    inputs = make_inputs()
    next(iter(inputs[2].values()))["success"] = success
    with pytest.raises(ValueError, match="boolean success"):
        summarize_cells(*inputs)


@pytest.mark.parametrize("extra", ["generation", "judgment", "score_without_generation"])
def test_unknown_artifact_keys_and_scores_without_generation_are_rejected(extra):
    inputs = make_inputs()
    if extra == "generation":
        inputs[1]["unknown"] = {"generation": "fixture"}
    elif extra == "judgment":
        inputs[2]["unknown"] = {"success": False}
    else:
        del inputs[1][next(iter(inputs[2]))]
    with pytest.raises(ValueError, match="Unexpected generation/judgment keys"):
        summarize_cells(*inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_level", True),
        ("source_level", 4),
        ("target_level", 0),
        ("layer", True),
        ("layer", 99),
    ],
)
def test_invalid_cell_metadata_is_rejected(field, value):
    inputs = make_inputs()
    inputs[0][0][field] = value
    with pytest.raises(ValueError, match="Invalid source level, target level, or layer"):
        summarize_cells(*inputs)


def test_missing_group_id_is_rejected():
    inputs = make_inputs()
    del inputs[-1][inputs[0][0]["behavior_id"]]
    with pytest.raises(ValueError, match="group ID"):
        summarize_cells(*inputs)


@pytest.mark.parametrize("layers", [[-1, -1], [True], ["0"]])
def test_invalid_layer_configuration_is_rejected(layers):
    inputs = list(make_inputs())
    inputs[3] = layers
    with pytest.raises(ValueError, match="Layers"):
        summarize_cells(*inputs)
