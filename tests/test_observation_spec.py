import pytest

from wsp.io import checkpoint_hashes, file_hash
from wsp.observation_spec import (
    generated_observation,
    normalize_capture_options,
    observation_layout,
)

MODEL = {
    "num_hidden_layers": 4,
    "hidden_size": 8,
    "intermediate_size": 12,
    "num_attention_heads": 2,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
}


def test_full_arrays_have_token_axes_and_explicit_validity_masks():
    plan = observation_layout(
        [3, 5],
        {
            "aggregation": "tokens",
            "hidden_states": [0, 4],
            "mlp_act": [2],
            "attn_layers": [3],
            "token_logprobs": True,
            "topk": 3,
        },
        MODEL,
    )
    shapes = {key: value["shape"] for key, value in plan["arrays"].items()}
    assert shapes["hidden_4"] == [2, 5, 8]
    assert shapes["mlp_2"] == [2, 5, 12]
    assert shapes["attention_3"] == [2, 2, 5, 5]
    assert shapes["topk_logprobs"] == [2, 5, 3]
    assert shapes["logprob_valid"] == shapes["valid_tokens"] == [2, 5]


def test_output_limit_is_exact_and_never_silently_averages():
    config = {"aggregation": "tokens", "hidden_states": [0]}
    amount = observation_layout([10], config, MODEL)["estimated_bytes"]
    observation_layout([10], {**config, "max_output_bytes": amount}, MODEL)
    with pytest.raises(ValueError, match="nothing was silently averaged"):
        observation_layout([10], {**config, "max_output_bytes": amount - 1}, MODEL)


def test_attention_restrictions_and_generation_summary_shapes():
    for config, lengths in [
        ({"attn_layers": [0]}, [10]),
        ({"attn_layers": [3]}, [2049]),
        ({"attn_layers": [3]}, [10, 10, 10]),
        ({"topk": 51}, [10]),
    ]:
        with pytest.raises(ValueError):
            observation_layout(lengths, config, MODEL)
    plan = observation_layout(
        [10, 20],
        {
            "aggregation": "generation",
            "hidden_states": [1],
            "mlp_act": [1],
        },
        MODEL,
    )
    assert plan["arrays"]["hidden_1_prompt_last"]["shape"] == [2, 8]
    assert plan["arrays"]["mlp_1_answer_mean"]["shape"] == [2, 12]


def test_token_defaults_are_full_observations_and_unknown_fields_fail():
    assert normalize_capture_options({}, "tokens")["aggregation"] == "tokens"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_capture_options({"hidden_state": [1]}, "tokens")
    with pytest.raises(ValueError, match="options object"):
        normalize_capture_options(None, "tokens")


def test_generated_answer_capture_uses_exact_first_8192_sampled_tokens():
    tokens, span = generated_observation("item", 0, [10000, 10001], list(range(9000)), True)
    assert tokens == [10000, 10001] + list(range(8192))
    assert span["answer_tokens"] == 8192
    assert span["full_generated_tokens"] == 9000
    assert span["prompt_tokens"] == 2
    assert span["correct"]


def test_observation_checkpoint_hashes_identify_actual_shards(tmp_path):
    with pytest.raises(ValueError, match="without weight shards"):
        checkpoint_hashes(tmp_path)
    shard = tmp_path / "model.safetensors-00001-of-00001.safetensors"
    shard.write_bytes(b"first checkpoint")
    first = checkpoint_hashes(tmp_path)
    assert first == {shard.name: file_hash(shard)}
    shard.write_bytes(b"another checkpoint")
    assert checkpoint_hashes(tmp_path) != first
