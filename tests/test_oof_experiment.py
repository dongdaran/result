"""No-network grouped OOF integration using harmless, synthetic fixture data."""

import json
import shutil
from collections import Counter
from copy import deepcopy

import pytest

from hb_steering import experiment
from hb_steering.storage import atomic_bytes, fingerprint, read_journal
from hb_steering.vectors import load_vectors, save_vectors


@pytest.fixture
def project_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_environment", lambda *_: None)

    def build(counts=(29, 189, 82)):
        root = tmp_path
        config = {
            "dataset": {"splits_path": "data/splits/splits.json", "n_folds": 5},
            "models": {
                "model_1": {"hf_id": "fake/target", "revision": "a" * 40, "dtype": "float32"}
            },
            "runtime": {"device": "cpu", "max_new_tokens": 16},
            "steering": {"layers": [-1], "coefficient": 0.5, "normalize": False},
            "judge": {"hf_id": "fake/judge", "revision": "b" * 40},
        }
        records, labels, group_ids = [], {}, {}
        held_out = {fold: [] for fold in range(1, 6)}
        for level, count in enumerate(counts, 1):
            sizes = [count // 5] * 5
            for index in range(count % 5):
                sizes[index + 1] += 1
            index = 0
            for fold, size in enumerate(sizes, 1):
                for _ in range(size):
                    ident = f"L{level}-{index:03d}"
                    shared = fold == 1 and size >= 2 and index < 2
                    records.append(
                        {
                            "behavior_id": ident,
                            "behavior": f"Safe fixture {ident}:level:{level}",
                            "context": f"Shared safe fixture L{level}" if shared else "",
                            "level": level,
                            "index": index,
                            "held_out_fold": fold,
                        }
                    )
                    labels[ident] = level
                    group_ids[ident] = f"L{level}-shared" if shared else f"group-{ident}"
                    held_out[fold].append(ident)
                    index += 1
        all_ids = set(labels)
        benign = [
            {"benign_id": f"safe{index:02d}", "prompt": f"benign {index}"} for index in range(20)
        ]
        saved = {
            "schema_version": 2,
            "fingerprint": "synthetic-dataset-identity",
            "harmbench": {
                "n_folds": 5,
                "group_ids": group_ids,
                "folds": [
                    {
                        "fold": fold,
                        "fit": sorted(all_ids - set(held_out[fold])),
                        "test": sorted(held_out[fold]),
                    }
                    for fold in range(1, 6)
                ],
            },
            "benign": {
                "fit": [row["benign_id"] for row in benign[:12]],
                "dev": [row["benign_id"] for row in benign[12:16]],
                "test": [row["benign_id"] for row in benign[16:]],
            },
        }
        prepared = [records, labels, benign, saved]
        monkeypatch.setattr(experiment, "read_prepared", lambda *_: deepcopy(prepared))
        config_path = root / "configs/experiment.toml"
        source_path = root / "src/hb_steering/fixture.py"
        for path, contents in (
            (config_path, "# Synthetic config snapshot\n"),
            (source_path, "# Synthetic source version 1\n"),
            (root / "uv.lock", "# Synthetic dependency lock version 1\n"),
            (root / config["dataset"]["splits_path"], json.dumps(saved)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)

        def setup(*, fold=None, limit=None, run_id="test-oof", actual_config=None):
            return experiment.setup_run(
                root,
                deepcopy(config if actual_config is None else actual_config),
                config_path,
                "model_1",
                run_id,
                fold,
                limit,
            )

        return {
            "root": root,
            "config": config,
            "prepared": prepared,
            "setup": setup,
            "source_path": source_path,
        }

    return build


class FakeRunner:
    hf_id = "fake/target"
    revision = "a" * 40
    hidden_size = 2
    layers = [None, None]

    def __init__(self):
        self.activation_inputs = []
        self.generation_calls = Counter()

    def activation_means(self, prompts, layers):
        import torch

        self.activation_inputs.append(list(prompts))
        levels = [
            0 if prompt.startswith("benign ") else int(prompt.rsplit(":", 1)[1])
            for prompt in prompts
        ]
        indices = [
            0 if prompt.startswith("benign ") else int(prompt.split(":level:")[0].rsplit("-", 1)[1])
            for prompt in prompts
        ]
        return {
            layer: torch.tensor([sum(levels) / len(levels), sum(indices) / len(indices)])
            for layer in layers
        }

    def generate(self, prompt, steering=None):
        source = 0 if steering is None else round(steering["vector"][0].item())
        self.generation_calls[(prompt, source)] += 1
        return f"fixture generation|{source}"


class FakeJudge:
    def __init__(self):
        self.calls = Counter()

    def judge(self, behavior, generation):
        source = int(generation.rsplit("|", 1)[1])
        self.calls[(behavior["behavior_id"], source)] += 1
        return behavior["index"] % 2 == 0 if source == 0 else source == behavior["level"]


def cells_by_key(report):
    return {
        (row["source_level"], row["layer"], row["target_level"]): row for row in report["cells"]
    }


def write_synthetic_journals(run, *, skip_folds=(), success=None):
    """Write complete valid records without loading a target or judge model."""
    success = success or (lambda job: job["fold"] == 1)
    for fold in run["all_fold_runs"]:
        if fold["fold"] in skip_folds:
            continue
        generations, judgments = [], []
        for job in fold["jobs"]:
            text = f"Safe synthetic generation {job['key']}"
            common = {
                **job,
                "run_fingerprint": fold["fingerprint"],
                "generation_sha256": fingerprint(text),
            }
            generations.append({**common, "generation": text})
            judgments.append({**common, "success": success(job)})
        for path, rows in (
            (fold["directory"] / "generations/responses.jsonl", generations),
            (fold["directory"] / "metrics/judgments.jsonl", judgments),
        ):
            atomic_bytes(path, "".join(json.dumps(row) + "\n" for row in rows).encode())


def test_setup_builds_all_folds_and_fold_selection_reuses_one_frozen_manifest(project_factory):
    project = project_factory()
    first = project["setup"](fold=1)
    manifest_path = project["root"] / "outputs/test-oof/manifest.json"
    frozen = manifest_path.read_bytes()
    second = project["setup"](fold=4)
    whole = project["setup"]()
    assert manifest_path.read_bytes() == frozen
    assert first["fingerprint"] == second["fingerprint"] == whole["fingerprint"]
    assert [run["fold"] for run in first["all_fold_runs"]] == [1, 2, 3, 4, 5]
    assert [run["fold"] for run in first["fold_runs"]] == [1]
    assert [run["fold"] for run in second["fold_runs"]] == [4]
    assert len(whole["fold_runs"]) == 5
    assert first["all_fold_runs"][3]["fingerprint"] == second["fold_runs"][0]["fingerprint"]
    assert len({run["fingerprint"] for run in whole["all_fold_runs"]}) == 5


@pytest.mark.parametrize("change", ["limit", "config", "data", "source", "lock", "splits"])
def test_changed_run_inputs_require_new_id(project_factory, change):
    project = project_factory()
    project["setup"]()
    kwargs = {}
    if change == "limit":
        kwargs["limit"] = 1
    elif change == "config":
        changed = deepcopy(project["config"])
        changed["steering"]["coefficient"] = 0.25
        kwargs["actual_config"] = changed
    elif change == "data":
        project["prepared"][3]["fingerprint"] = "different-synthetic-dataset"
    else:
        path = {
            "source": project["source_path"],
            "lock": project["root"] / "uv.lock",
            "splits": project["root"] / project["config"]["dataset"]["splits_path"],
        }[change]
        path.write_text(path.read_text() + "\nchanged fixture\n")
    with pytest.raises(ValueError, match="new --run-id"):
        project["setup"](**kwargs)
    assert project["setup"](run_id="fresh-oof", **kwargs)["all_fold_runs"]


def test_all_fit_levels_exclude_the_entire_held_out_fold_and_shared_groups(project_factory):
    project = project_factory()
    run = project["setup"]()
    all_ids = {row["behavior_id"] for row in run["records"]}
    evaluations = Counter()
    for fold in run["all_fold_runs"]:
        fit = {row["behavior_id"] for row in fold["fit"]}
        test = {row["behavior_id"] for row in fold["evaluation"]}
        assert fit == all_ids - test
        assert {run["labels"][ident] for ident in test} == {1, 2, 3}
        assert {run["labels"][ident] for ident in fit} == {1, 2, 3}
        assert not {run["group_ids"][ident] for ident in fit} & {
            run["group_ids"][ident] for ident in test
        }
        for job in fold["jobs"]:
            assert job["fold"] == fold["fold"]
            assert job["key"].startswith(f"fold:{fold['fold']}:")
        evaluations.update(test)
    assert set(evaluations) == all_ids
    assert set(evaluations.values()) == {1}


def test_five_fold_fake_model_flow_extracts_generates_scores_and_pools(project_factory):
    pytest.importorskip("torch")
    project = project_factory(counts=(10, 10, 10))
    run = project["setup"]()
    vector_paths, vector_values = set(), set()
    for fold in run["all_fold_runs"]:
        runner, judge = FakeRunner(), FakeJudge()
        payload = experiment.extract(fold, runner)
        assert payload["fold"] == fold["fold"]
        assert payload["run_fingerprint"] == fold["fingerprint"]
        vector_paths.add(experiment._vector_path(fold))
        vector_values.add(tuple(payload["vectors"][1][1].tolist()))
        used_prompts = {prompt for prompts in runner.activation_inputs[1:] for prompt in prompts}
        assert used_prompts == {experiment.behavior_prompt(row) for row in fold["fit"]}
        assert not used_prompts & {experiment.behavior_prompt(row) for row in fold["evaluation"]}
        generations = experiment.generate(fold, runner)
        judgments = experiment.score(fold, judge)
        assert len(generations) == len(judgments) == len(fold["jobs"])
        assert set(runner.generation_calls.values()) == {1}
        assert set(judge.calls.values()) == {1}
        # Completed fold artifacts are reused without re-extracting or re-generating.
        activation_count = len(runner.activation_inputs)
        generation_calls, judge_calls = runner.generation_calls.copy(), judge.calls.copy()
        experiment.extract(fold, runner)
        experiment.generate(fold, runner)
        experiment.score(fold, judge)
        assert len(runner.activation_inputs) == activation_count
        assert runner.generation_calls == generation_calls and judge.calls == judge_calls
        assert experiment.summarize(fold)["complete"] is True
    assert len(vector_paths) == len(vector_values) == 5
    report = experiment.summarize_oof(run)
    assert report["complete"] is report["full_oof"] is True
    assert report["folds_expected"] == report["folds_completed"] == [1, 2, 3, 4, 5]
    assert report["expected_jobs"] == report["generated_jobs"] == report["scored_jobs"] == 120
    assert len(report["cells"]) == 12
    for cell in report["cells"]:
        assert cell["n_expected"] == cell["n_scored"] == 10
        if cell["source_level"]:
            assert cell["asr"] == float(cell["source_level"] == cell["target_level"])
            assert cell["delta_asr"] == cell["asr"] - cell["baseline_asr"]
    for filename in ("summary.json", "asr.csv", "asr.md"):
        assert (run["directory"] / "metrics" / filename).is_file()


def test_vector_from_another_fold_is_rejected(project_factory):
    pytest.importorskip("torch")
    run = project_factory(counts=(10, 10, 10))["setup"]()
    first, second = run["all_fold_runs"][:2]
    experiment.extract(first, FakeRunner())
    other_path = experiment._vector_path(second)
    other_path.parent.mkdir(parents=True)
    shutil.copyfile(experiment._vector_path(first), other_path)
    with pytest.raises(ValueError, match="different run"):
        experiment.extract(second, FakeRunner())


def test_vector_explicit_fold_metadata_is_validated(project_factory):
    pytest.importorskip("torch")
    fold = project_factory(counts=(10, 10, 10))["setup"]()["all_fold_runs"][0]
    experiment.extract(fold, FakeRunner())
    path = experiment._vector_path(fold)
    payload = load_vectors(path)
    payload["fold"] = 99
    save_vectors(payload, path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="[Ff]old"):
        experiment.extract(fold, FakeRunner())


def test_300_behaviors_pool_to_full_level_denominators_and_weight_uneven_folds(project_factory):
    run = project_factory()["setup"]()
    write_synthetic_journals(run)
    report = experiment.summarize_oof(run)
    assert report["complete"] is report["full_oof"] is True
    assert report["expected_jobs"] == report["scored_jobs"] == 1200
    assert report["evaluation_scope"] == "pooled_oof"
    assert report["ci_method"] == "wilson_95_iid_binomial_diagnostic"
    assert "overlapping fit folds" in report["ci_limitation"]
    expected_counts = {1: 29, 2: 189, 3: 82}
    for cell in report["cells"]:
        assert (
            cell["n_expected"]
            == cell["n_generated"]
            == cell["n_scored"]
            == expected_counts[cell["target_level"]]
        )
        assert cell["n_groups"] == expected_counts[cell["target_level"]] - 1
    cell = cells_by_key(report)[(1, -1, 1)]
    assert cell["n_success"] == 5
    assert cell["asr"] == pytest.approx(5 / 29)
    fold_asrs = [
        cells_by_key(experiment.summarize(fold))[(1, -1, 1)]["asr"] for fold in run["all_fold_runs"]
    ]
    assert sum(fold_asrs) / len(fold_asrs) == 0.2
    assert cell["asr"] != sum(fold_asrs) / len(fold_asrs)
    assert cell["asr_ci_low"] < cell["asr"] < cell["asr_ci_high"]


def test_missing_fold_keeps_full_plan_but_withholds_pooled_asr_and_ci(project_factory):
    run = project_factory()["setup"](fold=1)
    write_synthetic_journals(run, skip_folds={5})
    report = experiment.summarize_oof(run)
    assert report["folds_expected"] == [1, 2, 3, 4, 5]
    assert report["folds_completed"] == [1, 2, 3, 4]
    assert report["expected_jobs"] == 1200
    assert report["generated_jobs"] == report["scored_jobs"] < report["expected_jobs"]
    assert report["complete"] is report["full_oof"] is False
    for cell in report["cells"]:
        assert cell["complete"] is False
        assert cell["n_scored"] < cell["n_expected"]
        assert cell["asr"] is cell["asr_ci_low"] is cell["asr_ci_high"] is None
        assert cell["baseline_asr"] is cell["delta_asr"] is None


def test_completed_limited_run_is_explicitly_not_full_oof(project_factory):
    run = project_factory()["setup"](limit=1)
    write_synthetic_journals(run)
    report = experiment.summarize_oof(run)
    assert report["complete"] is True
    assert report["full_oof"] is False
    assert report["limit_per_level"] == 1
    assert report["expected_jobs"] == report["scored_jobs"] == 60
    assert all(cell["n_expected"] == 5 for cell in report["cells"])


def test_pooled_summary_rejects_cross_fold_judgment_metadata(project_factory):
    run = project_factory()["setup"]()
    write_synthetic_journals(run)
    fold = run["all_fold_runs"][0]
    path = fold["directory"] / "metrics/judgments.jsonl"
    judgments = read_journal(path)
    next(iter(judgments.values()))["fold"] = 2
    atomic_bytes(path, "".join(json.dumps(row) + "\n" for row in judgments.values()).encode())
    with pytest.raises(ValueError, match="Judgment provenance/schema"):
        experiment.summarize_oof(run)


@pytest.mark.parametrize("fold", [0, 6, True, 1.5])
def test_unknown_or_noninteger_selected_fold_is_rejected(project_factory, fold):
    with pytest.raises(ValueError, match="Select --fold"):
        project_factory()["setup"](fold=fold)


def make_rule_transfer(project, monkeypatch, *, external=True, limit=None):
    config = deepcopy(project["config"])
    config["judge"] = {"method": "rule", "version": "gcg_refusal_v1"}
    config["evaluation"] = {"over_refusal": True}
    config["external"] = {
        "enabled": external,
        "labels_path": "data/annotations/external_severity.csv",
    }
    harmful = [
        {
            "behavior_id": f"jbb_harmful:{level}",
            "behavior": f"Synthetic external instruction {level}",
            "context": "",
            "functional_category": "standard",
            "level": level,
            "dataset": "jailbreakbench_original",
        }
        for level in (1, 2, 3)
    ]
    benign = [
        {
            "behavior_id": f"jbb_benign:{index}",
            "behavior": f"Synthetic external safe question {index}",
            "context": "",
            "level": 0,
            "dataset": "jailbreakbench_benign",
        }
        for index in range(5)
    ]
    if external:
        from hb_steering import external as external_module

        monkeypatch.setattr(external_module, "read_external", lambda *_: (harmful, benign))
        for path in (
            project["root"] / "data/processed/external/manifest.json",
            project["root"] / config["external"]["labels_path"],
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen external identity")
    return project["setup"](actual_config=config, limit=limit)


class RuleTransferRunner(FakeRunner):
    def generate(self, prompt, steering=None):
        source = 0 if steering is None else round(steering["vector"][0].item())
        self.generation_calls[(prompt, source)] += 1
        return "I can’t assist with this request." if source == 2 else "Synthetic useful response."


def test_final_transfer_fit_and_safe_evaluation_are_disjoint(project_factory, monkeypatch):
    run = make_rule_transfer(project_factory(counts=(10, 10, 10)), monkeypatch)
    transfer = run["transfer_run"]
    assert {r["behavior_id"] for r in transfer["fit"]} == {
        r["behavior_id"] for r in run["records"]
    }
    assert not {r["behavior_id"] for r in transfer["evaluation"]} & {
        r["behavior_id"] for r in transfer["fit"]
    }
    xstest_ids = {
        r["behavior_id"].removeprefix("xstest_safe:")
        for r in transfer["evaluation"]
        if r["dataset"] == "xstest_safe"
    }
    assert xstest_ids == {"safe16", "safe17", "safe18", "safe19"}
    assert not xstest_ids & {r["benign_id"] for r in transfer["benign_fit"]}
    assert len(transfer["jobs"]) == (3 + 4 + 5) * 4
    assert all("fold" not in job and job["key"].startswith("transfer:") for job in transfer["jobs"])


def test_rule_transfer_runs_once_reports_complements_and_resumes(project_factory, monkeypatch):
    pytest.importorskip("torch")
    from hb_steering.evaluate import HarmBenchJudge

    monkeypatch.setattr(HarmBenchJudge, "__init__", lambda *_: pytest.fail("No classifier weights"))
    run = make_rule_transfer(project_factory(counts=(10, 10, 10)), monkeypatch)
    transfer, runner = run["transfer_run"], RuleTransferRunner()
    payload = experiment.extract(transfer, runner)
    assert "fold" not in payload
    used = {prompt for prompts in runner.activation_inputs[1:] for prompt in prompts}
    assert used == {experiment.behavior_prompt(r) for r in transfer["fit"]}
    assert not used & {experiment.behavior_prompt(r) for r in transfer["evaluation"]}
    experiment.generate(transfer, runner)
    experiment.score(transfer)
    report = experiment.summarize_transfer(transfer)
    assert report["complete"] is report["full_transfer"] is True
    assert report["fit_counts"] == {"1": 10, "2": 10, "3": 10}
    assert len(report["cells"]) == 12 and len(report["over_refusal_cells"]) == 8
    for row in report["cells"]:
        assert row["asr_rule"] == float(row["source_level"] != 2)
    for row in report["over_refusal_cells"]:
        assert row["over_refusal"] == 1 - row["asr_rule"] == float(row["source_level"] == 2)
        assert row["n_expected"] == (4 if row["dataset"] == "xstest_safe" else 5)
    before = runner.generation_calls.copy()
    experiment.extract(transfer, runner)
    experiment.generate(transfer, runner)
    experiment.score(transfer)
    assert runner.generation_calls == before
    for filename in ("summary.json", "asr_rule.csv", "asr_rule.md", "over_refusal.csv", "over_refusal.md"):
        assert (transfer["directory"] / "metrics" / filename).is_file()


def test_transfer_incomplete_scores_are_not_counted_as_refusals(project_factory, monkeypatch):
    pytest.importorskip("torch")
    transfer = make_rule_transfer(project_factory(counts=(10, 10, 10)), monkeypatch)["transfer_run"]
    runner = RuleTransferRunner()
    experiment.extract(transfer, runner)
    experiment.generate(transfer, runner)
    report = experiment.summarize_transfer(transfer)
    assert report["complete"] is report["full_transfer"] is False
    assert all(row["asr_rule"] is None for row in report["cells"])
    assert all(row["over_refusal"] is None for row in report["over_refusal_cells"])


def test_rule_score_resume_rejects_tampered_rule_identity(project_factory, monkeypatch):
    pytest.importorskip("torch")
    transfer = make_rule_transfer(project_factory(counts=(10, 10, 10)), monkeypatch)["transfer_run"]
    runner = RuleTransferRunner()
    experiment.extract(transfer, runner)
    experiment.generate(transfer, runner)
    experiment.score(transfer)
    path = transfer["directory"] / "metrics/judgments.jsonl"
    rows = list(read_journal(path).values())
    rows[0]["evaluator"]["sha256"] = "wrong identity"
    atomic_bytes(path, "".join(json.dumps(row) + "\n" for row in rows).encode())
    with pytest.raises(ValueError, match="frozen evaluator"):
        experiment.score(transfer)


def test_safe_only_transfer_and_limited_fit_are_explicit(project_factory, monkeypatch):
    run = make_rule_transfer(project_factory(counts=(10, 10, 10)), monkeypatch, external=False, limit=1)
    transfer = run["transfer_run"]
    assert len(transfer["fit"]) == len(transfer["benign_fit"]) == 3
    assert len(transfer["evaluation"]) == 3
    assert {job["target_level"] for job in transfer["jobs"]} == {0}
    report = experiment.summarize_transfer(transfer)
    assert report["cells"] == [] and len(report["over_refusal_cells"]) == 4
    assert report["full_transfer"] is False and report["limit_per_level"] == 1


def test_lockless_run_snapshots_project_and_rejects_dependency_changes(project_factory, monkeypatch):
    project = project_factory()
    (project["root"] / "uv.lock").unlink()
    project_file = project["root"] / "pyproject.toml"
    project_file.write_text('[project]\nname = "fixture"\n')

    class Distribution:
        metadata = {"Name": "fixture-dependency"}
        version = "1.0"

    monkeypatch.setattr(experiment.metadata, "distributions", lambda: [Distribution()])
    project["setup"]()
    output = project["root"] / "outputs/test-oof"
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["dependency_lock_sha256"] is None
    assert saved["installed_dependencies"] == {"fixture-dependency": "1.0"}
    assert (output / "pyproject.toml").read_bytes() == project_file.read_bytes()
    project["setup"]()
    Distribution.version = "2.0"
    with pytest.raises(ValueError, match="new --run-id"):
        project["setup"]()
    Distribution.version = "1.0"
    project_file.write_text(project_file.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="new --run-id"):
        project["setup"]()
