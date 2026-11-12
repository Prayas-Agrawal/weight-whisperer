import pytest

from wsp.evaluate import EngineOptions, engine_options, render_prompt


def test_missing_engine_settings_do_not_silently_change_inference():
    with pytest.raises(KeyError):
        engine_options({"sampling": {"max_tokens": 32768}})


def test_compiled_profile_is_explicit_and_does_not_change_sampling():
    config = {
        "sampling": {"max_tokens": 32768, "temperature": 0.6},
        "engine": {"name": "compiled", "max_num_seqs": 64, "enforce_eager": False},
    }
    assert engine_options(config) == EngineOptions("compiled", 64, False)
    assert config["sampling"] == {"max_tokens": 32768, "temperature": 0.6}


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"name": "bad", "max_num_seqs": 0, "enforce_eager": True},
        {"name": "bad", "max_num_seqs": True, "enforce_eager": True},
        {"name": "bad", "max_num_seqs": 32, "enforce_eager": "false"},
    ],
)
def test_invalid_profiles_do_not_silently_fall_back(value):
    with pytest.raises(ValueError):
        engine_options({"engine": value})


def test_shared_prompt_renderer_keeps_scored_instruction_and_custom_messages():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            }
            assert messages[0]["role"] == "user"
            return messages[0]["content"]

    assert render_prompt({"problem": "1+1?"}, Tokenizer(), True) == (
        "1+1?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    assert (
        render_prompt({"problem": "Exact custom message", "custom_prompt": True}, Tokenizer(), True)
        == "Exact custom message"
    )
