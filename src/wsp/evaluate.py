"""Evaluate one frozen checkpoint with fixed per-problem seeds."""

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wsp.grading import grade, summarize
from wsp.io import (
    checkpoint_hashes,
    file_hash,
    item_seed,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from wsp.observation_spec import generated_observation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM


@dataclass(frozen=True)
class EngineOptions:
    name: str
    max_num_seqs: int
    enforce_eager: bool


def require_full_reasoning_budget(sampling: dict) -> None:
    if type(sampling.get("max_tokens")) is not int or sampling["max_tokens"] != 32768:
        raise ValueError(
            "Target generations require max_tokens=32768, including development probes. "
            "Use fewer questions or samples for speed; do not shorten per-answer reasoning."
        )


def validate_generation_purpose(config: dict) -> None:
    purpose = config.get("purpose", "measurement")
    if purpose not in {"measurement", "runtime_smoke"}:
        raise ValueError("Generation purpose must be measurement or runtime_smoke.")
    if purpose != "runtime_smoke":
        require_full_reasoning_budget(config["sampling"])


def engine_options(config: dict) -> EngineOptions:
    value = config["engine"]
    if not isinstance(value, dict) or set(value) != {"name", "max_num_seqs", "enforce_eager"}:
        raise ValueError("An engine profile requires name, max_num_seqs, and enforce_eager.")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("Engine name must be a nonempty string.")
    count = value["max_num_seqs"]
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 256:
        raise ValueError("max_num_seqs must be an integer from 1 to 256.")
    if not isinstance(value["enforce_eager"], bool):
        raise ValueError("enforce_eager must be a boolean.")
    return EngineOptions(value["name"], count, value["enforce_eager"])


def create_engine(config: dict) -> "LLM":
    engine = engine_options(config)
    batched_tokens = config.get("max_num_batched_tokens", 4096)
    if type(batched_tokens) is not int or batched_tokens < engine.max_num_seqs:
        raise ValueError("max_num_batched_tokens must be an integer covering max_num_seqs.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from vllm import LLM

    return LLM(
        model=str(Path(config["model"]).resolve()),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=34816,
        max_num_seqs=engine.max_num_seqs,
        max_num_batched_tokens=batched_tokens,
        enforce_eager=engine.enforce_eager,
        enable_prefix_caching=False,
        seed=config["sampling"]["generation_seed"],
        limit_mm_per_prompt={"image": 0, "video": 0},
    )


def render_prompt(problem: dict, tokenizer, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    problem["problem"]
                    if problem.get("custom_prompt", False)
                    else problem["problem"]
                    + "\nPlease reason step by step, "
                    + "and put your final answer within \\boxed{}."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def run(
    config: dict,
    llm: "LLM | None" = None,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> dict:
    validate_generation_purpose(config)
    engine = engine_options(config)
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model_path = Path(config["model"]).resolve()
    dataset_path = Path(config["dataset"]).resolve()
    output = Path(config["output"]).resolve()
    protocol = config["sampling"]
    problems = read_jsonl(dataset_path)
    if not problems or len({p["id"] for p in problems}) != len(problems):
        raise ValueError("Evaluation requires nonempty, unique problem IDs.")
    print(f"Model: {model_path}\nDataset: {dataset_path}\nOutput: {output}", flush=True)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = [render_prompt(p, tokenizer, protocol["enable_thinking"]) for p in problems]
    parameters = [
        SamplingParams(
            n=protocol["samples_per_problem"],
            temperature=protocol["temperature"],
            top_p=protocol["top_p"],
            top_k=protocol["top_k"],
            presence_penalty=protocol["presence_penalty"],
            max_tokens=protocol["max_tokens"],
            seed=item_seed(p["id"], protocol["generation_seed"]),
        )
        for p in problems
    ]
    if llm is None:
        llm = create_engine(config)
    generated = llm.generate(prompts, parameters, use_tqdm=False)
    if len(generated) != len(problems):
        raise RuntimeError("Inference returned the wrong number of problems.")
    records, texts, observations, spans = [], [], [], []
    for problem, response in zip(problems, generated, strict=True):
        if len(response.outputs) != protocol["samples_per_problem"]:
            raise RuntimeError("Inference returned an incomplete sample set.")
        for index, sample in enumerate(response.outputs):
            row = {
                "id": problem["id"],
                "sample": index,
                "tokens": len(sample.token_ids),
                "finish_reason": sample.finish_reason,
                "thinking_closed": "</think>" in sample.text,
                **(
                    {"correct": None, "extracted": None, "grading_status": "ungraded_custom_prompt"}
                    if problem.get("custom_prompt", False)
                    else grade(sample.text, problem["answer"])
                ),
            }
            records.append(row)
            texts.append({**row, "problem": problem["problem"], "text": sample.text})
            if config.get("activations") is not None:
                if not response.prompt_token_ids or not sample.token_ids:
                    raise RuntimeError(
                        "Generation activation capture requires actual sampled tokens."
                    )
                sequence, span = generated_observation(
                    problem["id"],
                    index,
                    list(response.prompt_token_ids),
                    list(sample.token_ids),
                    row["correct"],
                )
                observations.append(sequence)
                spans.append(span)
    summary = {
        "config": config,
        "model_checkpoint_sha256": checkpoint_hashes(model_path),
        "capability_measurement": config.get("purpose", "measurement") != "runtime_smoke",
        "resolved_engine": asdict(engine),
        "dataset_sha256": file_hash(dataset_path),
        "metrics": summarize(records),
    }
    write_jsonl(output / "generations.jsonl", texts)
    write_jsonl(output / "per_sample.jsonl", records)
    if observations:
        from wsp.capture import run as capture

        options = config["activations"]
        layers = options["layers"]
        captured = capture(
            {
                "model": str(model_path),
                "physical_gpu": config["physical_gpu"],
                "token_ids": observations,
                "generation_spans": spans,
                "max_texts": 512,
                "max_input_tokens": 34816,
                "max_total_tokens": sum(map(len, observations)),
                "aggregation": "generation",
                "hidden_states": layers,
                "mlp_act": [layer - 1 for layer in layers if layer > 0]
                if options.get("mlp")
                else [],
                "output": str(output / "activations"),
            }
        )
        captured["generation_file_sha256"] = file_hash(output / "generations.jsonl")
        write_json(output / "activations/meta.json", captured)
        summary["activations"] = {
            "folder": str(output / "activations"),
            "samples": len(observations),
            "method": captured["generation_method"],
        }
    write_json(output / "summary.json", summary)
    print(summary["metrics"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.config))


if __name__ == "__main__":
    main()
