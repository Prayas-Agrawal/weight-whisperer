"""Explicit tensor layout, token coordinates and bounded output sizes for model observations."""

import math

import numpy as np


def layer_indices(values: object, maximum: int, name: str) -> list[int]:
    if (
        not isinstance(values, list)
        or any(type(i) is not int or not 0 <= i <= maximum for i in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be distinct integer indices from 0 to {maximum}.")
    return values


def observation_layout(lengths: list[int], config: dict, model: dict) -> dict:
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("Observations require nonempty token sequences.")
    count, width = len(lengths), max(lengths)
    aggregation = config.get("aggregation", "mean")
    if not isinstance(aggregation, str) or aggregation not in {"mean", "tokens", "generation"}:
        raise ValueError("aggregation must be mean, tokens or generation.")
    layers = model["num_hidden_layers"]
    hidden = layer_indices(config.get("hidden_states", []), layers, "hidden_states")
    mlp = layer_indices(config.get("mlp_act", []), layers - 1, "mlp_act")
    attention = layer_indices(config.get("attn_layers", []), layers - 1, "attn_layers")
    if any(model["layer_types"][layer] != "full_attention" for layer in attention):
        raise ValueError("Attention maps are available only for full-attention layers.")
    if attention and (count > 2 or width > 2048):
        raise ValueError("Attention capture permits at most 2 texts of 2048 tokens.")
    topk = config.get("topk", 0)
    if type(topk) is not int or not 0 <= topk <= 50:
        raise ValueError("topk must be an integer from 0 to 50.")
    if type(config.get("token_logprobs", False)) is not bool:
        raise ValueError("token_logprobs must be a boolean.")
    if aggregation == "mean":
        tail = config.get("tail_tokens", 128)
        if type(tail) is not int or not 1 <= tail <= config.get("max_input_tokens", 32768):
            raise ValueError("Invalid tail_tokens.")
    arrays = {
        "input_ids": {"shape": [count, width], "dtype": "int64"},
        "valid_tokens": {"shape": [count, width], "dtype": "bool"},
    }

    def add(key, feature_width):
        if aggregation == "generation":
            for suffix in ["prompt_last", "answer_mean"]:
                arrays[f"{key}_{suffix}"] = {
                    "shape": [count, feature_width],
                    "dtype": "float32",
                }
        else:
            shape = (
                [count, width, feature_width]
                if aggregation == "tokens"
                else [
                    count,
                    feature_width,
                ]
            )
            arrays[key] = {"shape": shape, "dtype": "float32"}

    for layer in hidden:
        add(f"hidden_{layer}", model["hidden_size"])
    for layer in mlp:
        add(f"mlp_{layer}", model["intermediate_size"])
    for layer in attention:
        arrays[f"attention_{layer}"] = {
            "shape": [count, model["num_attention_heads"], width, width],
            "dtype": "float32",
        }
    if config.get("token_logprobs", False):
        arrays["token_logprobs"] = {"shape": [count, width], "dtype": "float32"}
        arrays["logprob_valid"] = {"shape": [count, width], "dtype": "bool"}
    if topk:
        arrays["topk_ids"] = {"shape": [count, width, topk], "dtype": "int64"}
        arrays["topk_logprobs"] = {"shape": [count, width, topk], "dtype": "float32"}
    estimated = sum(
        math.prod(row["shape"]) * np.dtype(row["dtype"]).itemsize + 4096 for row in arrays.values()
    )
    limit = config.get("max_output_bytes", 6_000_000_000)
    if type(limit) is not int or limit < 1 or estimated > limit:
        raise ValueError(
            f"Requested observation arrays require about {estimated} bytes; limit is {limit}. "
            "Use fewer layers/texts or shorter contexts; nothing was silently averaged."
        )
    return {
        "aggregation": aggregation,
        "arrays": arrays,
        "estimated_bytes": estimated,
        "hidden_layers": hidden,
        "mlp_layers": mlp,
        "attention_layers": attention,
        "topk": topk,
    }


def normalize_capture_options(options: dict, default_aggregation: str) -> dict:
    if not isinstance(options, dict):
        raise ValueError("return must be an observation-options object.")
    known = {
        "hidden_states",
        "mlp_act",
        "attn_layers",
        "tail_tokens",
        "aggregation",
        "token_logprobs",
        "topk",
    }
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"Unknown observation options: {sorted(unknown)}")
    return {
        "hidden_states": options.get("hidden_states", []),
        "mlp_act": options.get("mlp_act", []),
        "attn_layers": options.get("attn_layers", []),
        "aggregation": options.get("aggregation", default_aggregation),
        "tail_tokens": options.get("tail_tokens", 128),
        "token_logprobs": options.get("token_logprobs", False),
        "topk": options.get("topk", 0),
    }


def generated_observation(
    identifier: str,
    sample: int,
    prompt_ids: list[int],
    answer_ids: list[int],
    correct: bool | None,
) -> tuple[list[int], dict]:
    if not prompt_ids or not answer_ids:
        raise ValueError("Generation observations require prompt and sampled-answer tokens.")
    observed_answer = answer_ids[:8192]
    return prompt_ids + observed_answer, {
        "id": identifier,
        "sample": sample,
        "correct": correct,
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": len(observed_answer),
        "full_generated_tokens": len(answer_ids),
    }
