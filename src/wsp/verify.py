"""Verify the released results from their complete per-sample records, without a GPU."""

import gzip
import json
import math
from pathlib import Path

from wsp.config import ROOT
from wsp.grading import summarize
from wsp.io import file_hash, read_json

EXPECTED = {(f"math500/test/{i}", sample) for i in range(500) for sample in range(4)}


def checked_file(root: Path, info: dict, label: str) -> Path:
    path = (root / info["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != info["sha256"]:
        raise ValueError(f"Release file checksum mismatch: {label}")
    return path


def verify_arm(root: Path, arm: dict, label: str) -> float:
    with gzip.open(checked_file(root, arm["scores"], label), "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 2000 or {(row["id"], row["sample"]) for row in rows} != EXPECTED:
        raise ValueError(f"Missing, duplicate or wrong sample IDs: {label}")
    if any(
        type(row["correct"]) is not bool
        or not 1 <= row["tokens"] <= 32768
        or row["finish_reason"] not in {"stop", "length"}
        or (row["finish_reason"] == "length" and row["tokens"] != 32768)
        for row in rows
    ):
        raise ValueError(f"Invalid grade or generation status: {label}")
    metrics = summarize(rows)
    if (
        metrics != arm["metrics"]
        or sum(row["correct"] for row in rows) != arm["correct_samples"]
        or sum(row["correct"] and row["finish_reason"] == "stop" for row in rows)
        != arm["correct_and_finished"]
        or not math.isclose(round(100 * metrics["accuracy"], 2), arm["accuracy_percentage"])
    ):
        raise ValueError(f"Per-sample records disagree with the published metrics: {label}")
    return arm["accuracy_percentage"]


def verify_result(root: Path = ROOT) -> dict:
    result = read_json(root / "results/math500.json")
    benchmark = result["benchmark"]
    if (
        benchmark["questions"],
        benchmark["samples_per_question"],
        benchmark["samples_per_model"],
    ) != (500, 4, 2000):
        raise ValueError("The published results must cover 500 questions with four samples each.")
    for relative, expected in result["prompt_sha256"].items():
        checked_file(root, {"path": relative, "sha256": expected}, relative)
    original = verify_arm(root, result["original"], "original")
    runs = {}
    for label, run in result["runs"].items():
        for key in ["delta", "manifest"]:
            checked_file(root, run["edit"][key], f"{label} {key}")
        accuracy = verify_arm(root, run, label)
        if not math.isclose(round(accuracy - original, 2), run["gain_percentage_points"]):
            raise ValueError(f"Incorrect percentage-point gain: {label}")
        research = run["research"]
        if (
            research["research_guidance_messages"] != 0
            or research["model_selected_final"] is not True
        ):
            raise ValueError(f"{label} must be an autonomous run with a model-selected final.")
        runs[label] = {
            "accuracy_percent": accuracy,
            "gain_percentage_points": run["gain_percentage_points"],
        }
    if not runs:
        raise ValueError("No published runs were found.")
    return {
        "verified": True,
        "questions": 500,
        "samples_per_model": 2000,
        "original_accuracy_percent": original,
        "runs": runs,
    }


def main() -> None:
    print(json.dumps(verify_result(), indent=2))


if __name__ == "__main__":
    main()
