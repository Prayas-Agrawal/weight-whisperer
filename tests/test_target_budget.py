import pytest

from wsp.config import settings
from wsp.editor import generation_sampling, practice_selection
from wsp.evaluate import require_full_reasoning_budget, run


@pytest.mark.parametrize("tokens", [0, -1, 8192, 32767, 32769, True, 2.5, "32768"])
def test_shortened_or_invalid_answer_budgets_are_rejected(tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        generation_sampling(settings(), {"model": "original", "max_tokens": tokens}, 2)


def test_full_capacity_is_available_without_reducing_reasoning():
    config = settings()
    assert generation_sampling(config, {"model": "original"}, 512)["max_tokens"] == 32768
    assert generation_sampling(config, {"model": "both", "n": 2}, 128)["max_tokens"] == 32768
    with pytest.raises(ValueError, match="capacity"):
        generation_sampling(config, {"model": "both", "n": 2}, 129)
    with pytest.raises(ValueError, match="positive integer"):
        generation_sampling(config, {"model": "original", "n": True}, 1)


def test_direct_evaluation_cannot_bypass_the_budget_guard():
    with pytest.raises(ValueError, match="fewer questions"):
        run({"sampling": {"max_tokens": 8192}})
    require_full_reasoning_budget({"max_tokens": 32768})


def test_only_mounted_practice_ids_are_scored():
    practice = {"id-1": {"id": "id-1", "problem": "1+1?", "answer": "2"}}
    assert practice_selection({"model": "original", "problems": ["id-1"]}, practice) == [
        practice["id-1"]
    ]
    with pytest.raises(ValueError, match="mounted practice"):
        practice_selection({"model": "original", "problems": ["benchmark-id"]}, practice)
    custom = practice_selection({"model": "edited", "prompts": ["Exact message"]}, practice)
    assert custom[0]["answer"] is None and custom[0]["custom_prompt"]
