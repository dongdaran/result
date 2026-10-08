import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hb_steering import cli, experiment, models
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
    with patch.object(
        cli, "read_prepared", return_value=([{}], {"a": 1}, [{}], {"fingerprint": "x"})
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
        assert request.call_count == 1  # The bundled judge is already pinned.
    pinned, _ = load_config(project)
    assert pinned["models"]["model_1"]["revision"] == revision
    assert "# Fill these later." in path.read_text()
    with patch.object(cli.urllib.request, "urlopen", side_effect=AssertionError("Already pinned")):
        cli.pin_models(pinned, path)


def test_pin_models_has_no_partial_write_on_remote_failure(project):
    config, path = load_config(project)
    text = path.read_text().replace('hf_id = ""', 'hf_id = "example/model"', 1)
    text = text.replace(config["judge"]["revision"], "")
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


def test_run_unloads_target_before_judge(project):
    config, path = configured(project)
    calls = []
    run = {"directory": project / "outputs/x/model_1"}

    class Runner:
        def __init__(self, *_):
            calls.append("load")

        def close(self):
            calls.append("close")

    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", return_value=run),
        patch.object(models, "HFRunner", Runner),
        patch.object(experiment, "extract", side_effect=lambda *_: calls.append("extract")),
        patch.object(experiment, "generate", side_effect=lambda *_: calls.append("generate")),
        patch.object(experiment, "score", side_effect=lambda *_: calls.append("score")),
        patch.object(
            experiment,
            "summarize",
            return_value={"complete": True, "scored_jobs": 24, "expected_jobs": 24},
        ),
    ):
        assert cli.main(["--root", str(project), "run", "--model", "model_1", "--run-id", "x"]) == 0
    assert calls == ["load", "extract", "generate", "close", "score"]


def test_smoke_caps_generation_and_keeps_judge_unloaded(project):
    config, path = configured(project)
    run = {"directory": project / "outputs/x/model_1"}
    captured = {}

    class Runner:
        def __init__(self, _, runtime):
            captured["runtime"] = runtime

        def close(self):
            captured["closed"] = True

    def setup(root, actual, source, model, ident, split, limit):
        captured.update(split=split, limit=limit)
        return run

    with (
        patch.object(cli, "load_config", return_value=(config, path)),
        patch.object(experiment, "setup_run", side_effect=setup),
        patch.object(models, "HFRunner", Runner),
        patch.object(experiment, "extract"),
        patch.object(experiment, "generate"),
        patch.object(experiment, "score", side_effect=AssertionError("No judge in smoke")),
    ):
        assert (
            cli.main(["--root", str(project), "smoke", "--model", "model_1", "--run-id", "x"]) == 0
        )
    assert captured["runtime"]["max_new_tokens"] == 16
    assert captured["split"] == "dev" and captured["limit"] == 2 and captured["closed"]
    assert config["runtime"]["max_new_tokens"] == 512
