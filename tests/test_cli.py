import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hb_steering import cli, evaluate, experiment, models
from hb_steering.config import load_config


@pytest.fixture
def project(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    text = (root / "configs/experiment.toml").read_text()
    (tmp_path / "configs/experiment.toml").write_text(text)
    return tmp_path


def test_unselected_model_fails_before_loading_weights(project, capsys):
    with patch.object(models, "HFRunner", side_effect=AssertionError("Must not load a model")):
        assert cli.main(["--root", str(project), "run", "--model", "model_1", "--run-id", "x"]) == 2
    assert "Set models.model_1.hf_id" in capsys.readouterr().err


def test_download_and_check_use_stdlib_pipeline(project, capsys):
    with patch.object(cli, "prepare_dataset", return_value={"ready": True}) as prepare:
        assert cli.main(["--root", str(project), "download", "--resplit"]) == 0
        assert prepare.call_args.kwargs == {"resplit": True}
    config, path = load_config(project)
    config["external"]["enabled"] = False
    with patch.object(cli, "load_config", return_value=(config, path)), patch.object(
        cli,
        "read_prepared",
        return_value=([{}], {"a": 1}, [{}], {"fingerprint": "x", "harmbench": {"n_folds": 5}}),
    ):
        assert cli.main(["--root", str(project), "check"]) == 0
    assert '"status": "ok"' in capsys.readouterr().out


def test_pin_models_preserves_comments_and_skips_existing_pins(project):
    config, path = load_config(project)
    text = path.read_text().replace('hf_id = ""', 'hf_id = "example/model"', 1)
    path.write_text(text)
    config, path = load_config(project)
    revision = "a" * 40
    with patch.object(
        cli.urllib.request,
        "urlopen",
        return_value=io.BytesIO(json.dumps({"sha": revision}).encode()),
    ) as request:
        cli.pin_models(config, path)
        assert request.call_count == 1  # Rule evaluation does not resolve a judge model.
    pinned, _ = load_config(project)
    assert pinned["models"]["model_1"]["revision"] == revision
    assert "# Fill these later." in path.read_text()
    with patch.object(cli.urllib.request, "urlopen", side_effect=AssertionError("Already pinned")):
        cli.pin_models(pinned, path)


def test_pin_models_has_no_partial_write_on_remote_failure(project):
    config, path = load_config(project)
    text = path.read_text().replace('hf_id = ""', 'hf_id = "example/model"', 2)
    path.write_text(text)
    config, path = load_config(project)
    with patch.object(
        cli.urllib.request,
        "urlopen",
        side_effect=[io.BytesIO(json.dumps({"sha": "a" * 40}).encode()), OSError("offline")],
    ):
        with pytest.raises(OSError):
            cli.pin_models(config, path)
    assert path.read_text() == text


def configured(project):
    config, path = load_config(project)
    config = copy.deepcopy(config)
    config["models"]["model_1"].update(hf_id="example/model", revision="a" * 40)
    return config, path


def make_run_context(project, config, fold_ids=(1, 2, 3, 4, 5)):
    return {
        "directory": project / "outputs/x/model_1",
        "config": config,
        "model_key": "model_1",
        "fold_runs": [
            {
                "directory": project / f"outputs/x/model_1/fold_{fold}",
                "fold": fold,
                "jobs": [{"key": f"fold:{fold}:job"}],
            }
            for fold in fold_ids
        ],
    }


def test_run_reuses_one_target_then_unloads_before_one_judge(project):
    config, path = configured(project)
    # Legacy classifier runs still release target weights before loading the judge.
    config["judge"] = {"hf_id": "cais/HarmBench-Llama-2-13b-cls", "revision": "b" * 40}
    config["evaluation"]["over_refusal"] = False
    calls = []
    run = make_run_context(project, config)

    class Runner:
        def __init__(self, *_):
            calls.append("target_load")

        def close(self):
            calls.append("target_close")

    class Judge:
        def __init__(self, *_):
            self.runner = object()
            calls.append("judge_load")

        def close(self):
            calls.append("judge_close")

    report = {"complete": True, "scored_jobs": 120, "expected_jobs": 120, "full_oof": True}
    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", return_value=run),
        patch.object(models, "HFRunner", Runner),
        patch.object(evaluate, "HarmBenchJudge", Judge),
        patch.object(
            experiment,
            "extract",
            side_effect=lambda fold, *_: calls.append(f"extract{fold['fold']}"),
        ),
        patch.object(
            experiment,
            "generate",
            side_effect=lambda fold, *_: calls.append(f"generate{fold['fold']}"),
        ),
        patch.object(
            experiment,
            "_validated_records",
            side_effect=lambda fold: ({fold["jobs"][0]["key"]: {}}, {}),
        ),
        patch.object(experiment, "_environment"),
        patch.object(
            experiment, "score", side_effect=lambda fold, *_: calls.append(f"score{fold['fold']}")
        ),
        patch.object(experiment, "summarize", return_value=report),
        patch.object(experiment, "summarize_oof", return_value=report),
    ):
        assert cli.main(["--root", str(project), "run", "--model", "model_1", "--run-id", "x"]) == 0
    assert calls == ["target_load"] + [
        phase + str(fold) for fold in range(1, 6) for phase in ("extract", "generate")
    ] + ["target_close", "judge_load"] + [f"score{fold}" for fold in range(1, 6)] + ["judge_close"]


def test_smoke_caps_generation_and_keeps_judge_unloaded(project):
    config, path = configured(project)
    captured = {}

    class Runner:
        def __init__(self, _, runtime):
            captured["runtime"] = runtime

        def close(self):
            captured["closed"] = True

    def setup(root, actual, source, model, ident, fold, limit):
        captured.update(fold=fold, limit=limit)
        return make_run_context(project, actual, (fold,))

    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", side_effect=setup),
        patch.object(models, "HFRunner", Runner),
        patch.object(experiment, "extract"),
        patch.object(experiment, "generate"),
        patch.object(cli, "_score_folds", side_effect=AssertionError("No judge in smoke")),
    ):
        assert (
            cli.main(["--root", str(project), "smoke", "--model", "model_1", "--run-id", "x"]) == 0
        )
    assert captured["runtime"]["max_new_tokens"] == 16
    assert captured["fold"] == 1 and captured["limit"] == 2 and captured["closed"]
    assert config["runtime"]["max_new_tokens"] == 512


def test_completed_target_folds_resume_without_model_loading(project):
    from hb_steering.storage import append_record

    config, _ = configured(project)
    run = make_run_context(project, config)
    for fold in run["fold_runs"]:
        path = fold["directory"] / "vectors/directions.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        append_record(fold["directory"] / "generations/responses.jsonl", fold["jobs"][0])
    with (
        patch.object(models, "HFRunner", side_effect=AssertionError("Must reuse completed target")),
        patch.object(experiment, "extract") as extract,
        patch.object(experiment, "generate") as generate,
    ):
        cli._run_target_folds(run, "run")
    assert extract.call_count == generate.call_count == 5


def test_single_fold_success_can_leave_full_pool_incomplete(project, capsys):
    config, path = configured(project)
    run = make_run_context(project, config, (2,))
    pooled = {"complete": False, "scored_jobs": 24, "expected_jobs": 120, "full_oof": False}
    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", return_value=run) as setup,
        patch.object(experiment, "summarize", return_value={"complete": True}),
        patch.object(experiment, "summarize_oof", return_value=pooled),
    ):
        assert (
            cli.main(
                [
                    "--root",
                    str(project),
                    "summarize",
                    "--model",
                    "model_1",
                    "--run-id",
                    "x",
                    "--fold",
                    "2",
                ]
            )
            == 0
        )
    assert setup.call_args.args[5] == 2
    assert "complete=False; full_oof=False" in capsys.readouterr().out


def test_legacy_split_flag_is_rejected():
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args(["run", "--model", "model_1", "--run-id", "x", "--split", "dev"])
    assert error.value.code == 2


def test_transfer_context_is_selected_only_for_all_fold_commands():
    fold, transfer = {"fold": 1}, {"scope": "transfer"}
    run = {"fold_runs": [fold], "transfer_run": transfer, "selected_fold": None}
    assert cli._selected_runs(run) == [fold, transfer]
    run["selected_fold"] = 1
    assert cli._selected_runs(run) == [fold]


def test_rule_scoring_never_loads_classifier_or_environment(project):
    config, _ = configured(project)
    run = make_run_context(project, config, (1,))
    with (
        patch.object(evaluate, "HarmBenchJudge", side_effect=AssertionError("No classifier")),
        patch.object(experiment, "_environment", side_effect=AssertionError("No GPU environment")),
        patch.object(experiment, "_validated_records", return_value=({"fold:1:job": {}}, {})),
        patch.object(experiment, "score") as score,
    ):
        cli._score_folds(run)
    assert isinstance(score.call_args.args[1], evaluate.RuleJudge)


def test_transfer_command_runs_only_final_context(project):
    config, path = configured(project)
    run = make_run_context(project, config)
    transfer = {"directory": run["directory"] / "transfer"}
    run["transfer_run"] = transfer
    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", return_value=run),
        patch.object(cli, "_run_target_folds") as target,
        patch.object(cli, "_score_folds") as score,
        patch.object(experiment, "summarize_transfer", return_value={"complete": True}),
        patch.object(experiment, "summarize_oof", side_effect=AssertionError("No OOF work")),
    ):
        assert cli.main(["--root", str(project), "transfer", "--model", "model_1", "--run-id", "x"]) == 0
    assert cli._selected_runs(target.call_args.args[0]) == [transfer]
    assert cli._selected_runs(score.call_args.args[0]) == [transfer]
