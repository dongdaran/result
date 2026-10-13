"""No-network integration tests for resumable generation, judging, and 3x3 ASR."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from hb_steering import experiment
from hb_steering.evaluate import JudgeOutputError
from hb_steering.storage import atomic_bytes, fingerprint, read_journal
from hb_steering.vectors import load_vectors, save_vectors


@pytest.fixture
def run(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(experiment, "_environment", lambda runner, path: None)
    fit = [
        {
            "behavior_id": f"fit{level}",
            "behavior": f"fit{level}",
            "context": "",
            "test_level": level,
        }
        for level in (1, 2, 3)
    ]
    evaluation = [
        {
            "behavior_id": f"eval{level}{suffix}",
            "behavior": f"eval{level}{suffix}",
            "context": "",
            "test_level": level,
        }
        for level in (1, 2, 3)
        for suffix in ("a", "b")
    ]
    records = fit + evaluation
    labels = {row["behavior_id"]: row["test_level"] for row in records}
    config = {
        "models": {"model_1": {"hf_id": "fake/target", "revision": "a" * 40, "dtype": "float32"}},
        "runtime": {"device": "cpu"},
        "steering": {"layers": [-1], "coefficient": 0.5, "normalize": False},
        "judge": {"hf_id": "fake/judge", "revision": "b" * 40},
    }
    return {
        "root": tmp_path,
        "directory": tmp_path / "outputs/run/model_1",
        "config": config,
        "model_key": "model_1",
        "records": records,
        "labels": labels,
        "fit": fit,
        "evaluation": evaluation,
        "benign_fit": [{"benign_id": "safe1", "prompt": "benign"}],
        "jobs": experiment.plan_jobs(evaluation, labels, [-1]),
        "fingerprint": "original-run-identity",
    }


class FakeRunner:
    hf_id = "fake/target"
    revision = "a" * 40
    hidden_size = 2
    layers = [None, None]

    def __init__(self, fail_once=()):
        self.calls = Counter()
        self.fail_once = set(fail_once)

    def activation_means(self, prompts, layers):
        import torch

        values = [0 if prompt == "benign" else int(prompt[-1]) for prompt in prompts]
        value = sum(values) / len(values)
        return {layer: torch.tensor([value, value + 0.5]) for layer in layers}

    def generate(self, prompt, steering=None):
        source = 0 if steering is None else round(steering["vector"][0].item())
        key = f"baseline:{prompt}" if source == 0 else f"steered:{source}:-1:{prompt}"
        self.calls[key] += 1
        if key in self.fail_once:
            self.fail_once.remove(key)
            raise RuntimeError("temporary generation failure")
        return f"{prompt}|{source}"


class FakeJudge:
    def __init__(self, fail_once=(), invalid_once=()):
        self.calls = Counter()
        self.fail_once = set(fail_once)
        self.invalid_once = set(invalid_once)

    def judge(self, behavior, generation):
        self.calls[generation] += 1
        if generation in self.fail_once:
            self.fail_once.remove(generation)
            raise JudgeOutputError("unexpected classifier output")
        if generation in self.invalid_once:
            self.invalid_once.remove(generation)
            return 0
        ident, source = generation.split("|")
        return ident.endswith("a") if source == "0" else int(source) == behavior["test_level"]


def populate_generations(run, runner=None):
    runner = runner or FakeRunner()
    experiment.extract(run, runner)
    return experiment.generate(run, runner), runner


def rewrite_journal(path, rows):
    atomic_bytes(path, "".join(json.dumps(row) + "\n" for row in rows).encode())


def test_generation_retries_only_missing_job_then_reuses_completed_run(run):
    failed = "steered:2:-1:eval2b"
    runner = FakeRunner(fail_once={failed})
    experiment.extract(run, runner)
    with pytest.raises(RuntimeError, match="1 generation jobs failed"):
        experiment.generate(run, runner)
    path = run["directory"] / "generations/responses.jsonl"
    before = read_journal(path)
    assert len(before) == len(run["jobs"]) - 1
    assert failed not in before
    assert len(read_journal(run["directory"] / "logs/errors.jsonl")) == 1
    completed = experiment.generate(run, runner)
    assert len(completed) == len(run["jobs"]) == 24
    assert runner.calls[failed] == 2
    assert all(count == 1 for key, count in runner.calls.items() if key != failed)
    counts = runner.calls.copy()
    assert experiment.generate(run, runner) == completed
    assert runner.calls == counts
    assert len(read_journal(path)) == 24


def test_generation_resumes_after_torn_final_append(run):
    rows, runner = populate_generations(run)
    removed = next(reversed(rows))
    path = run["directory"] / "generations/responses.jsonl"
    rewrite_journal(path, [row for key, row in rows.items() if key != removed])
    with path.open("ab") as stream:
        stream.write(b'{"key":"interrupted')
    before = runner.calls.copy()
    result = experiment.generate(run, runner)
    assert len(result) == 24
    assert runner.calls[removed] == before[removed] + 1
    assert all(runner.calls[key] == count for key, count in before.items() if key != removed)


def test_judge_errors_and_nonboolean_results_remain_unscored_and_retry(run):
    populate_generations(run)
    judge = FakeJudge(fail_once={"eval1a|2"}, invalid_once={"eval3b|1"})
    with pytest.raises(RuntimeError, match="2 judge jobs failed"):
        experiment.score(run, judge)
    judgments = read_journal(run["directory"] / "metrics/judgments.jsonl")
    assert len(judgments) == 22
    assert "steered:2:-1:eval1a" not in judgments
    assert "steered:1:-1:eval3b" not in judgments
    errors = read_journal(run["directory"] / "logs/errors.jsonl")
    assert {row["error_type"] for row in errors.values()} == {"JudgeOutputError", "ValueError"}
    completed = experiment.score(run, judge)
    assert len(completed) == 24
    assert judge.calls["eval1a|2"] == judge.calls["eval3b|1"] == 2
    assert all(
        count == 1 for text, count in judge.calls.items() if text not in {"eval1a|2", "eval3b|1"}
    )
    counts = judge.calls.copy()
    experiment.score(run, judge)
    assert judge.calls == counts


def test_complete_summary_has_nine_transfer_cells_and_equal_baseline_denominators(run):
    populate_generations(run)
    experiment.score(run, FakeJudge())
    report = experiment.summarize(run)
    assert report["complete"] is True
    assert report["expected_jobs"] == report["generated_jobs"] == report["scored_jobs"] == 24
    baseline = [cell for cell in report["cells"] if cell["source_level"] == 0]
    transfer = [cell for cell in report["cells"] if cell["source_level"] != 0]
    assert len(baseline) == 3 and len(transfer) == 9
    assert {(cell["source_level"], cell["target_level"]) for cell in transfer} == {
        (source, target) for source in (1, 2, 3) for target in (1, 2, 3)
    }
    assert all(
        cell["n_expected"] == cell["n_generated"] == cell["n_scored"] == 2
        for cell in report["cells"]
    )
    assert all(cell["asr"] == 0.5 for cell in baseline)
    for cell in transfer:
        expected = float(cell["source_level"] == cell["target_level"])
        assert cell["asr"] == expected
        assert cell["baseline_asr"] == 0.5
        assert cell["delta_asr"] == expected - 0.5
    assert (run["directory"] / "metrics/asr.csv").is_file()
    assert (run["directory"] / "metrics/asr.md").is_file()


def test_incomplete_cells_have_null_asr_and_null_delta_with_incomplete_baseline(run):
    populate_generations(run)
    judge = FakeJudge(fail_once={"eval1a|0", "eval2a|3"})
    with pytest.raises(RuntimeError):
        experiment.score(run, judge)
    report = experiment.summarize(run)
    assert report["complete"] is False
    cells = {(cell["source_level"], cell["target_level"]): cell for cell in report["cells"]}
    assert cells[(0, 1)]["n_scored"] == 1
    assert cells[(0, 1)]["asr"] is None
    assert cells[(3, 2)]["n_scored"] == 1
    assert cells[(3, 2)]["asr"] is None
    assert cells[(3, 2)]["delta_asr"] is None
    assert cells[(1, 1)]["asr"] == 1.0
    assert cells[(1, 1)]["baseline_asr"] is None
    assert cells[(1, 1)]["delta_asr"] is None


def test_changed_scored_generation_is_rejected_even_with_recomputed_content_hash(run):
    rows, _runner = populate_generations(run)
    experiment.score(run, FakeJudge())
    changed = deepcopy(rows)
    changed["baseline:eval1a"]["generation"] = "changed text"
    changed["baseline:eval1a"]["generation_sha256"] = fingerprint("changed text")
    rewrite_journal(run["directory"] / "generations/responses.jsonl", changed.values())
    with pytest.raises(ValueError, match="Judgment provenance"):
        experiment.score(run, FakeJudge())
    with pytest.raises(ValueError, match="Judgment provenance"):
        experiment.summarize(run)


def test_modified_generation_without_matching_hash_is_rejected(run):
    rows, runner = populate_generations(run)
    rows["baseline:eval1a"]["generation"] = "unexpected mutation"
    rewrite_journal(run["directory"] / "generations/responses.jsonl", rows.values())
    with pytest.raises(ValueError, match="content checksum"):
        experiment.generate(run, runner)


def test_vector_provenance_survives_missing_sidecar_and_rejects_another_run(run):
    experiment.extract(run, FakeRunner())
    path = run["directory"] / "vectors/directions.pt"
    sidecar = path.with_suffix(".json")
    sidecar.unlink()
    loaded = experiment.extract(run, FakeRunner())
    assert loaded["run_fingerprint"] == run["fingerprint"]
    assert sidecar.is_file()
    sidecar.unlink()
    other_run = {**run, "fingerprint": "different-run-identity"}
    with pytest.raises(ValueError, match="different run"):
        experiment.extract(other_run, FakeRunner())
    assert not sidecar.exists()


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("model", "does not match"),
        ("layers", "layers differ"),
        ("source_levels", "levels/layers differ"),
        ("checksum", "checksum/run identity"),
    ),
)
def test_vector_model_layers_levels_and_file_checksum_are_checked(run, mutation, match):
    experiment.extract(run, FakeRunner())
    path = run["directory"] / "vectors/directions.pt"
    payload = load_vectors(path)
    if mutation == "model":
        payload["model"]["revision"] = "c" * 40
    elif mutation == "layers":
        payload["requested_layers"] = [0]
    elif mutation == "source_levels":
        del payload["vectors"][3]
        del payload["counts"]["levels"][3]
    else:
        payload["vectors"][1][1][0] += 0.1
    save_vectors(payload, path)
    with pytest.raises(ValueError, match=match):
        experiment.extract(run, FakeRunner())


def test_scoring_requires_complete_generation(run):
    runner = FakeRunner(fail_once={"baseline:eval1a"})
    experiment.extract(run, runner)
    with pytest.raises(RuntimeError):
        experiment.generate(run, runner)
    judge = FakeJudge()
    with pytest.raises(ValueError, match="Complete generation"):
        experiment.score(run, judge)
    assert not judge.calls


def test_resolved_vector_layers_must_match_the_loaded_decoder(run):
    experiment.extract(run, FakeRunner())
    path = run["directory"] / "vectors/directions.pt"
    payload = load_vectors(path)
    # Keep declared requested layer [-1] while moving self-consistent tensors
    # from the final block (1) to block 0. This must fail before generation.
    payload["layers"] = [0]
    payload["vectors"] = {level: {0: values[1]} for level, values in payload["vectors"].items()}
    save_vectors(payload, path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Resolved vector layers"):
        experiment.extract(run, FakeRunner())


def test_batched_generation_groups_conditions_and_resumes_missing_chunks(run):
    run["config"]["runtime"]["batch_size"] = 2

    class BatchRunner(FakeRunner):
        def generate_batch(self, prompts, steering=None):
            return [self.generate(prompt, steering) for prompt in prompts]

    runner = BatchRunner(fail_once={"baseline:eval1a"})
    experiment.extract(run, runner)
    with pytest.raises(RuntimeError, match="2 generation jobs failed"):
        experiment.generate(run, runner)
    path = run["directory"] / "generations/responses.jsonl"
    existing = read_journal(path)
    assert len(existing) == 22
    assert "baseline:eval1a" not in existing and "baseline:eval1b" not in existing
    before = runner.calls.copy()
    result = experiment.generate(run, runner)
    assert len(result) == 24
    for key in existing:
        assert runner.calls[key] == before[key]
    assert len(read_journal(run["directory"] / "logs/errors.jsonl")) == 2
