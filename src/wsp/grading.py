from collections import defaultdict

import numpy as np
from math_verify import LatexExtractionConfig, parse, verify


def last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    brace = start + len(r"\boxed")
    while brace < len(text) and text[brace].isspace():
        brace += 1
    if brace == len(text) or text[brace] != "{":
        return None
    depth = 1
    for index in range(brace + 1, len(text)):
        if text[index] == "{" and text[index - 1] != "\\":
            depth += 1
        elif text[index] == "}" and text[index - 1] != "\\":
            depth -= 1
        if depth == 0:
            return text[start : index + 1]
    return None


def grade(text: str, answer: str) -> dict:
    extraction = [LatexExtractionConfig()]
    gold = parse(r"\boxed{" + answer + "}", extraction_config=extraction)
    if not gold:
        raise ValueError(f"Unparseable reference answer: {answer!r}")
    boxed = last_boxed(text)
    if boxed is None:
        return {"correct": False, "extracted": None, "grading_status": "missing_box"}
    prediction = parse(boxed, extraction_config=extraction)
    if not prediction:
        return {"correct": False, "extracted": boxed, "grading_status": "unparseable_prediction"}
    return {
        "correct": bool(verify(gold, prediction)),
        "extracted": boxed,
        "grading_status": "graded",
    }


def summarize(records: list[dict], bootstrap_seed: int = 0) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation.")
    grouped = defaultdict(list)
    sample_counts = defaultdict(int)
    seen = set()
    for row in records:
        key = (row["id"], row["sample"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation observation: {key}")
        seen.add(key)
        sample_counts[row["id"]] += 1
        if row["correct"] is None:
            if row["grading_status"] != "ungraded_custom_prompt":
                raise ValueError("Missing correctness without an explicit ungraded custom prompt.")
        else:
            grouped[row["id"]].append(float(row["correct"]))
    counts = set(sample_counts.values())
    if len(counts) != 1:
        raise ValueError("Unequal sample counts would change the accuracy estimand.")
    means = np.array([np.mean(values) for values in grouped.values()])
    interval = None
    if len(means):
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "problems": len(sample_counts),
        "graded_problems": len(grouped),
        "ungraded_samples": sum(r["correct"] is None for r in records),
        "samples": len(records),
        "samples_per_problem": counts.pop(),
        "accuracy": float(means.mean()) if len(means) else None,
        "problem_bootstrap_95ci": interval,
        "cut_at_token_limit": float(np.mean([r["finish_reason"] == "length" for r in records])),
        "mean_tokens": float(np.mean([r["tokens"] for r in records])),
        "thinking_closed": float(np.mean([r["thinking_closed"] for r in records])),
        "missing_box": sum(r["grading_status"] == "missing_box" for r in records),
    }


def paired_comparison(base: list[dict], edited: list[dict], seed: int = 0) -> dict:
    left = {(r["id"], r["sample"]): r for r in base}
    right = {(r["id"], r["sample"]): r for r in edited}
    if len(left) != len(base) or len(right) != len(edited) or left.keys() != right.keys():
        raise ValueError("Paired comparisons require identical, unique problem/sample keys.")
    if not left:
        raise ValueError("Cannot compare empty evaluations.")
    grouped = defaultdict(list)
    ungraded = 0
    rescues = regressions = truncated_rescues = 0
    finished_differences = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["correct"] is None or b["correct"] is None:
            if a["correct"] is not None or b["correct"] is not None:
                raise ValueError("Paired items have inconsistent grading availability.")
            ungraded += 1
            continue
        delta = int(b["correct"]) - int(a["correct"])
        grouped[key[0]].append(delta)
        rescues += delta == 1
        regressions += delta == -1
        truncated_rescues += delta == 1 and a["finish_reason"] == "length"
        if a["finish_reason"] != "length":
            finished_differences.append(delta)
    if not grouped:
        return {
            "accuracy_difference": None,
            "ungraded_pairs": ungraded,
            "reason": "Custom prompts have no reference answers; no accuracy comparison computed.",
        }
    per_problem = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_problem, size=(5000, len(per_problem)), replace=True).mean(axis=1)
    return {
        "accuracy_difference": float(per_problem.mean()),
        "ungraded_pairs": ungraded,
        "paired_problem_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "rescued_samples": rescues,
        "regressed_samples": regressions,
        "rescues_previously_truncated": truncated_rescues,
        "originally_finished_samples": len(finished_differences),
        "difference_on_originally_finished": (
            float(np.mean(finished_differences)) if finished_differences else None
        ),
    }
