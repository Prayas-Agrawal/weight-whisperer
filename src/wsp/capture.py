"""Full or summarized teacher-forced observations, with explicit masks and token coordinates."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from wsp.io import checkpoint_hashes, read_json, write_json
from wsp.observation_spec import observation_layout

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def encode_inputs(config: dict, tokenizer) -> list[list[int]]:
    for key in ["enable_thinking", "add_generation_prompt"]:
        if key in config and type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    fields = [key for key in ["texts", "chats", "token_ids"] if key in config]
    if len(fields) != 1:
        raise ValueError("Supply exactly one of texts, chats or token_ids.")
    items = config[fields[0]]
    if not isinstance(items, list) or not 1 <= len(items) <= config.get("max_texts", 1024):
        raise ValueError("Invalid number of teacher-forced inputs.")
    encoded = []
    for item in items:
        if fields[0] == "texts":
            if not isinstance(item, str) or not item:
                raise ValueError("Teacher-forced texts must be nonempty strings.")
            tokens = tokenizer.encode(item)
        elif fields[0] == "chats":
            if not isinstance(item, list) or not item:
                raise ValueError("Each chat must be a nonempty message list.")
            for message in item:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(
                        "Chat observations accept text system/user/assistant messages."
                    )
            tokens = tokenizer.apply_chat_template(
                item,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=config.get("add_generation_prompt", False),
                enable_thinking=config.get("enable_thinking", True),
            )
        else:
            tokens = item
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or not 0 <= token < len(tokenizer) for token in tokens)
        ):
            raise ValueError("Invalid token IDs.")
        if len(tokens) > config.get("max_input_tokens", 32768):
            raise ValueError("A context exceeds max_input_tokens; no truncation was applied.")
        encoded.append(tokens)
    if sum(map(len, encoded)) > config.get("max_total_tokens", 524288):
        raise ValueError("Teacher-forced request exceeds the total input-token allowance.")
    return encoded


def run(
    config: dict,
    model: "PreTrainedModel | None" = None,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> dict:
    if model is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    encoded = encode_inputs(config, tokenizer)
    lengths = list(map(len, encoded))
    model_config = read_json(Path(config["model"]) / "config.json")["text_config"]
    plan = observation_layout(lengths, config, model_config)
    generation = config.get("generation_spans")
    if plan["aggregation"] == "generation":
        if not isinstance(generation, list) or len(generation) != len(encoded):
            raise ValueError("Generation observations require one span per sampled sequence.")
        for length, span in zip(lengths, generation, strict=True):
            if (
                type(span["prompt_tokens"]) is not int
                or not 1 <= span["prompt_tokens"] < length
                or span["answer_tokens"] != length - span["prompt_tokens"]
            ):
                raise ValueError("Invalid generation prompt/answer boundaries.")
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite observations: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < plan["estimated_bytes"] + 1024**3:
        raise RuntimeError("Insufficient disk space for the requested observation arrays.")
    torch.set_num_threads(4)
    if model is None:
        model = (
            AutoModelForImageTextToText.from_pretrained(
                config["model"],
                dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="eager" if plan["attention_layers"] else "sdpa",
            )
            .to("cuda")
            .eval()
        )
    backbone = model.model.language_model
    device = next(backbone.parameters()).device
    shapes = {key: value["shape"] for key, value in plan["arrays"].items()}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        folder = Path(temp)
        arrays = {
            key: np.lib.format.open_memmap(
                folder / f"{key}.npy",
                mode="w+",
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
            )
            for key, value in plan["arrays"].items()
        }
        for array in arrays.values():
            array[:] = 0
        index = 0
        seen = set()

        def capture_tensor(key, tensor):
            if key in seen:
                raise RuntimeError(f"Observation hook fired twice for {key}")
            seen.add(key)
            values = tensor[0].detach()
            if not torch.isfinite(values).all():
                raise RuntimeError(f"Nonfinite activation values at {key}")
            if plan["aggregation"] == "tokens":
                arrays[key][index, : lengths[index]] = values.float().cpu().numpy()
            elif plan["aggregation"] == "mean":
                tail = config.get("tail_tokens", 128)
                arrays[key][index] = values[-tail:].float().mean(dim=0).cpu().numpy()
            else:
                prompt = generation[index]["prompt_tokens"]
                arrays[f"{key}_prompt_last"][index] = values[prompt - 1].float().cpu().numpy()
                arrays[f"{key}_answer_mean"][index] = (
                    values[prompt:].float().mean(dim=0).cpu().numpy()
                )

        def output_hook(key):
            def hook(_module, _inputs, tensor):
                capture_tensor(key, tensor)

            return hook

        def input_hook(key):
            def hook(_module, inputs):
                capture_tensor(key, inputs[0])

            return hook

        def attention_hook(key):
            def hook(_module, _inputs, outputs):
                attention = outputs[1]
                if attention is None:
                    raise RuntimeError(
                        "Attention weights unavailable; eager attention is required."
                    )
                if not torch.isfinite(attention).all():
                    raise RuntimeError("Nonfinite attention weights.")
                length = lengths[index]
                arrays[key][index, :, :length, :length] = attention[0].float().cpu().numpy()
                seen.add(key)

            return hook

        handles = []
        expected = set()
        try:
            for layer in plan["hidden_layers"]:
                key = f"hidden_{layer}"
                module = backbone.embed_tokens if layer == 0 else backbone.layers[layer - 1]
                handles.append(module.register_forward_hook(output_hook(key)))
                expected.add(key)
            for layer in plan["mlp_layers"]:
                key = f"mlp_{layer}"
                handles.append(
                    backbone.layers[layer].mlp.down_proj.register_forward_pre_hook(input_hook(key))
                )
                expected.add(key)
            for layer in plan["attention_layers"]:
                key = f"attention_{layer}"
                handles.append(
                    backbone.layers[layer].self_attn.register_forward_hook(attention_hook(key))
                )
                expected.add(key)
            with torch.inference_mode():
                for index, ids in enumerate(encoded):
                    seen.clear()
                    length = len(ids)
                    arrays["input_ids"][index, :length] = ids
                    arrays["valid_tokens"][index, :length] = True
                    tokens = torch.tensor([ids], device=device)
                    result = backbone(input_ids=tokens, use_cache=False)
                    if seen != expected:
                        raise RuntimeError(f"Missing observation hooks: {expected - seen}")
                    if config.get("token_logprobs", False) or plan["topk"]:
                        for start in range(0, length, 128):
                            end = min(start + 128, length)
                            logits = model.lm_head(result.last_hidden_state[:, start:end]).float()
                            logprobs = torch.log_softmax(logits[0], dim=-1)
                            if not torch.isfinite(logprobs).all():
                                raise RuntimeError("Nonfinite target log-probabilities.")
                            if config.get("token_logprobs", False):
                                stop = min(end, length - 1)
                                if stop > start:
                                    targets = tokens[0, start + 1 : stop + 1]
                                    values = (
                                        logprobs[: stop - start]
                                        .gather(
                                            -1,
                                            targets[:, None],
                                        )
                                        .squeeze(-1)
                                    )
                                    arrays["token_logprobs"][index, start + 1 : stop + 1] = (
                                        values.cpu().numpy()
                                    )
                                    arrays["logprob_valid"][index, start + 1 : stop + 1] = True
                            if plan["topk"]:
                                values, indices = logprobs.topk(plan["topk"], dim=-1)
                                arrays["topk_ids"][index, start:end] = indices.cpu().numpy()
                                arrays["topk_logprobs"][index, start:end] = values.cpu().numpy()
                            del logits, logprobs
        finally:
            for handle in handles:
                handle.remove()
        for array in arrays.values():
            array.flush()
        metadata = {
            "model_path": str(Path(config["model"]).resolve()),
            "model_checkpoint_sha256": checkpoint_hashes(Path(config["model"])),
            "texts": len(encoded),
            "input_tokens": lengths,
            "arrays": shapes,
            "array_specs": plan["arrays"],
            "aggregation": plan["aggregation"],
            "estimated_bytes": plan["estimated_bytes"],
            "tail_tokens": config.get("tail_tokens", 128)
            if plan["aggregation"] == "mean"
            else None,
            "padding": "Right padded with zero; valid_tokens distinguishes real tokens.",
            "hidden_coordinates": (
                "hidden_0 is embedding output; hidden_L is after block L-1, BEFORE final norm."
            ),
            "mlp_coordinates": "mlp_L is the input to down_proj at block L.",
            "logprob_coordinates": (
                "token_logprobs[:,t] = log P(input_ids[t] | tokens before t). "
                "Position0 and padding are invalid; consult logprob_valid."
            ),
            "topk_coordinates": (
                "topk at position t predicts the NEXT token after input_ids[t]. "
                "Only positions with valid_tokens=True are meaningful."
            ),
            "attention_coordinates": "attention_L[b,head,query,key], with zero right padding.",
            "generation_spans": generation,
            "generation_method": (
                "Teacher-forced replay of exact sampled token IDs using the same checkpoint; "
                "not online vLLM internal-state capture."
                if plan["aggregation"] == "generation"
                else None
            ),
        }
        write_json(folder / "meta.json", metadata)
        arrays.clear()
        folder.rename(output)
    print({"texts": len(encoded), "aggregation": plan["aggregation"], "arrays": shapes}, flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.config))


if __name__ == "__main__":
    main()
