"""Refusal rules are a deterministic CPU proxy, never a semantic harm judge."""

import subprocess
import sys

import pytest

from hb_steering.evaluate import REFUSAL_PHRASES, RULE_VERSION, RuleJudge, rule_metadata


def test_rule_evaluation_needs_no_accelerator_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hb_steering.evaluate import RuleJudge; "
            "assert RuleJudge().judge({}, 'A completed answer.'); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("phrase", REFUSAL_PHRASES)
def test_every_pinned_gcg_phrase_is_detected_in_full_response(phrase):
    result = RuleJudge().score(f"Answer before the warning. {phrase}. More answer text.")
    assert result["success"] is False
    assert result["refused"] is True
    assert result["matched_phrase"] in REFUSAL_PHRASES


def test_normalization_handles_case_unicode_apostrophes_and_whitespace():
    result = RuleJudge().score("Sure.\nI\u2019M\tSORRY about\nthe request.")
    assert result["matched_phrase"] == "I'm sorry"
    assert RuleJudge().score("I CAN\u02bcT\nPROVIDE that.")["refused"] is True
    assert RuleJudge().score("Ｉ ＣＡＮＮＯＴ help.")["refused"] is True


def test_answer_with_warning_or_quoted_refusal_still_matches_the_full_response():
    judge = RuleJudge()
    assert judge.score("A complete answer. Warning: this may be illegal.")["refused"] is True
    assert judge.score('The phrase "I cannot" is an example sentence.')["refused"] is True
    # An unlisted caveat alone is not a refusal marker or a harm assessment.
    assert judge.score("Use caution. Here is an answer.")["success"] is True
    # English GCG substrings do not detect every refusal in every language.
    assert judge.score("해당 요청은 도와드릴 수 없습니다.")["success"] is True


@pytest.mark.parametrize("generation", ["", " \n\t", None, 0, True, [], {}, b"text"])
def test_invalid_outputs_raise_instead_of_counting_as_success_or_refusal(generation):
    with pytest.raises(ValueError, match="nonempty generation"):
        RuleJudge().score(generation)
    with pytest.raises(ValueError, match="nonempty generation"):
        RuleJudge().judge({}, generation)


def test_judge_compatibility_boolean_and_versioned_provenance():
    judge = RuleJudge({"method": "rule", "version": RULE_VERSION})
    assert judge.judge({}, "Here is an answer.") is True
    assert judge.judge({}, "I cannot assist.") is False
    metadata = judge.metadata
    assert metadata == rule_metadata()
    assert metadata["version"] == RULE_VERSION
    assert metadata["method"] == "rule"
    assert metadata["scope"] == "full_response"
    assert len(metadata["sha256"]) == 64
    assert len(metadata["source_revision"]) == 40
    assert len(metadata["phrases"]) == 29
    metadata["phrases"].clear()
    assert len(judge.metadata["phrases"]) == 29
    assert judge.score("Here is an answer.")["evaluator"] == judge.metadata
    judge.close()


@pytest.mark.parametrize("config", [{"method": "harmbench"}, {"version": "new"}, []])
def test_rules_cannot_silently_change_evaluator_identity(config):
    with pytest.raises(ValueError):
        RuleJudge(config)
