from pathlib import Path

import pytest

from hb_steering.config import load_config, require_judge


@pytest.fixture
def configured(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs/experiment.toml"
    target = tmp_path / "configs/experiment.toml"
    target.parent.mkdir()
    target.write_text(source.read_text())
    return tmp_path, target


def test_default_rule_scorer_needs_no_model_revision_or_gpu(configured):
    root, _ = configured
    config, _ = load_config(root)
    assert require_judge(config) == {"method": "rule", "version": "gcg_refusal_v1"}
    assert config["evaluation"]["over_refusal"] is True
    assert config["external"]["enabled"] is True
    assert config["external"]["source"] == "jailbreakbench_original"


@pytest.mark.parametrize(
    ("original", "replacement", "error"),
    [
        ('method = "rule"', 'method = "unknown"', "judge.method"),
        ('version = "gcg_refusal_v1"', 'version = "changed_rules"', "judge.version"),
        ("over_refusal = true", 'over_refusal = "true"', "evaluation.over_refusal"),
        ("enabled = true", "enabled = 1", "external.enabled"),
        (
            'source = "jailbreakbench_original"',
            'source = "all_jailbreakbench"',
            "external.source",
        ),
        (
            'revision = "886acc352a31533ffbcf4ef22c744658688086fc"',
            'revision = "main"',
            "external.revision",
        ),
        (
            'harmful_sha256 = "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1"',
            'harmful_sha256 = "not-a-hash"',
            "external.harmful_sha256",
        ),
        (
            'benign_sha256 = "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468"',
            'benign_sha256 = ""',
            "external.benign_sha256",
        ),
        (
            'labels_path = "data/annotations/external_severity.csv"',
            'labels_path = ""',
            "external.labels_path",
        ),
    ],
)
def test_invalid_scoring_protocol_and_external_pins_fail_early(
    configured, original, replacement, error
):
    root, path = configured
    path.write_text(path.read_text().replace(original, replacement))
    with pytest.raises(ValueError, match=error):
        load_config(root)


def classifier_config(path):
    text = path.read_text().split("[judge]")[0]
    text = text.replace("over_refusal = true", "over_refusal = false")
    text = text.replace("enabled = true", "enabled = false")
    path.write_text(
        text
        + '[judge]\nhf_id = "cais/HarmBench-Llama-2-13b-cls"\n'
        + f'revision = "{"a" * 40}"\n'
    )


def test_legacy_classifier_keeps_revision_requirement(configured):
    root, path = configured
    classifier_config(path)
    config, _ = load_config(root)
    assert require_judge(config)["revision"] == "a" * 40
    config["judge"]["revision"] = "main"
    with pytest.raises(ValueError, match="Pin judge.revision"):
        require_judge(config)


def test_classifier_cannot_be_used_for_one_minus_asr_rule(configured):
    root, path = configured
    classifier_config(path)
    path.write_text(path.read_text().replace("over_refusal = false", "over_refusal = true"))
    with pytest.raises(ValueError, match="over_refusal requires rule scoring"):
        load_config(root)


def test_classifier_cannot_be_mislabeled_as_external_asr_rule(configured):
    root, path = configured
    classifier_config(path)
    path.write_text(path.read_text().replace("enabled = false", "enabled = true"))
    with pytest.raises(ValueError, match="external.enabled requires rule scoring"):
        load_config(root)


def test_legacy_config_may_omit_new_evaluation_sections(configured):
    root, path = configured
    classifier_config(path)
    text = path.read_text()
    beginning, remainder = text.split("[evaluation]", 1)
    _, models = remainder.split("[models.model_1]", 1)
    path.write_text(beginning + "[models.model_1]" + models)
    config, _ = load_config(root)
    assert "evaluation" not in config and "external" not in config
    assert require_judge(config)["revision"] == "a" * 40
