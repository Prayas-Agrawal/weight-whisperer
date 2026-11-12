import pytest

from wsp.grading import grade, last_boxed, paired_comparison, summarize
from wsp.io import item_seed


def row(identifier, correct, sample=0, finish="stop"):
    return {
        "id": identifier,
        "sample": sample,
        "correct": correct,
        "finish_reason": finish,
        "tokens": 100,
        "thinking_closed": True,
        "grading_status": "graded",
    }


def test_last_box_handles_nested_braces():
    assert last_boxed(r"\boxed{0} then \boxed{\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"
    assert last_boxed(r"\boxed{42") is None
    assert last_boxed("42") is None
    assert last_boxed(r"\boxed{42} then \boxed{") is None


def test_equivalence_and_missing_answer():
    assert grade(r"\boxed{\frac{1}{2}}", "0.5")["correct"]
    assert not grade(r"\boxed{7}", "8")["correct"]
    assert grade("42", "42")["grading_status"] == "missing_box"


def test_accuracy_is_not_pass_at_k():
    result = summarize([row("a", True, 0), row("a", False, 1)])
    assert result["accuracy"] == 0.5
    assert result["samples_per_problem"] == 2
    assert result["problem_bootstrap_95ci"] == [0.5, 0.5]


def test_empty_duplicate_and_unequal_samples_fail():
    for records in [
        [],
        [row("a", True), row("a", False)],
        [
            row("a", True),
            row("a", True, 1),
            row("b", False),
        ],
    ]:
        with pytest.raises(ValueError):
            summarize(records)


def test_paired_statistics_preserve_problem_clusters_and_termination():
    before = [row("a", False, finish="length"), row("b", True)]
    after = [row("b", True), row("a", True)]
    result = paired_comparison(before, after)
    assert result["accuracy_difference"] == 0.5
    assert result["rescues_previously_truncated"] == 1
    assert result["difference_on_originally_finished"] == 0.0
    with pytest.raises(ValueError):
        paired_comparison(before, [row("c", True)])


def test_per_problem_seed_is_stable_and_distinct():
    assert item_seed("gsm8k/train/1", 0) == item_seed("gsm8k/train/1", 0)
    assert item_seed("gsm8k/train/1", 0) != item_seed("gsm8k/train/2", 0)


def test_custom_prompts_do_not_get_fake_accuracy():
    record = {
        **row("custom/a", None),
        "grading_status": "ungraded_custom_prompt",
    }
    result = summarize([record])
    assert result["accuracy"] is None
    assert result["graded_problems"] == 0
    assert result["ungraded_samples"] == 1
    assert paired_comparison([record], [record])["accuracy_difference"] is None
    result = summarize([record, row("scored", True)])
    assert result["accuracy"] == 1
    assert result["graded_problems"] == 1
