"""Resumable full MATH-500 x4 evaluation of one frozen candidate and its original."""

import argparse
import copy
import fcntl
import gzip
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from wsp.config import ROOT, resolved_models, run_name, settings
from wsp.evaluate import create_engine, render_prompt, require_full_reasoning_budget
from wsp.evaluate import run as evaluate
from wsp.grading import summarize
from wsp.io import (
    checkpoint_hashes,
    compress_file,
    file_hash,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from wsp.rebuild import published_edits, rebuild
from wsp.resources import require_gpu_headroom, require_headroom, stop_process_tree

ARMS = ("original", "edited")


def plan(rows: list[dict], profile: dict) -> list[dict]:
    if [row["id"] for row in rows] != [f"math500/test/{i}" for i in range(500)]:
        raise ValueError(
            "Evaluation requires all 500 pinned MATH questions exactly once, in order."
        )
    gpus = profile["physical_gpus"]
    count = profile["problems_per_batch"]
    if (
        not gpus
        or len(gpus) != len(set(gpus))
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
        or type(count) is not int
        or count < 1
        or profile["samples_per_problem"] != 4
        or count * 4 > profile["engine"]["max_num_seqs"]
    ):
        raise ValueError("Invalid GPU allocation or four-sample batch size.")
    return [
        {
            "index": number,
            "gpu": gpus[number % len(gpus)],
            "ids": [row["id"] for row in rows[start : start + count]],
        }
        for number, start in enumerate(range(0, 500, count))
    ]


def immutable(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Recorded evaluation settings changed: {path.name}")
    write_json(path, value)


def prepare(
    root: Path, name: str, research: str | None, gpus: list[int] | None, edit: str = "run1"
) -> dict:
    candidate = rebuild(root, research, edit)
    config = settings(root)
    profile = copy.deepcopy(config["evaluation"])
    if gpus is not None:
        profile["physical_gpus"] = gpus
    dataset = root / "runs/data/evaluation/math500.jsonl"
    expected_hash = read_json(root / "results/math500.json")["benchmark"]["dataset_sha256"]
    if file_hash(dataset) != expected_hash:
        raise ValueError("Evaluation inputs do not match the pinned full benchmark.")
    rows = read_jsonl(dataset)
    batches = plan(rows, profile)
    sampling = {**config["sampling"], "samples_per_problem": 4}
    require_full_reasoning_budget(sampling)
    base = Path(resolved_models(root)["target"]["path"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    longest = max(
        len(tokenizer.encode(render_prompt(row, tokenizer, True), add_special_tokens=False))
        for row in rows
    )
    if longest + 32768 > 34816:
        raise ValueError("A benchmark prompt cannot fit with its full answer allowance.")
    manifest = {
        "name": name,
        "published_edit": None if research else edit,
        "research_run": research,
        "dataset": str(dataset),
        "dataset_sha256": expected_hash,
        "questions": 500,
        "samples_per_arm": 2000,
        "sampling": sampling,
        "profile": profile,
        "batches": batches,
        "models": {
            arm: {"path": str(path), "sha256": checkpoint_hashes(path)}
            for arm, path in [("original", base), ("edited", candidate)]
        },
        "metric": "Mean accuracy over four samples per question, not pass@4.",
    }
    immutable(root / "runs/evaluation" / name / "manifest.json", manifest)
    return manifest


def request_config(root: Path, manifest: dict, arm: str, batch: dict) -> dict:
    output = root / "runs/evaluation" / manifest["name"] / arm / f"batch_{batch['index']:03d}"
    return {
        "purpose": "measurement",
        "model": manifest["models"][arm]["path"],
        "dataset": str(output / "questions.jsonl"),
        "output": str(output),
        "physical_gpu": batch["gpu"],
        "gpu_memory_utilization": manifest["profile"]["gpu_memory_utilization"],
        "engine": manifest["profile"]["engine"],
        "max_num_batched_tokens": manifest["profile"]["max_num_batched_tokens"],
        "sampling": manifest["sampling"],
    }


def validate_batch(
    summary: dict, rows: list[dict], config: dict, ids: list[str], hashes: dict
) -> None:
    require_full_reasoning_budget(config["sampling"])
    expected = {(identifier, sample) for identifier in ids for sample in range(4)}
    if (
        config["sampling"]["samples_per_problem"] != 4
        or summary["config"] != config
        or summary.get("capability_measurement") is not True
        or summary["model_checkpoint_sha256"] != hashes
        or summary["dataset_sha256"] != file_hash(Path(config["dataset"]))
        or len(rows) != len(expected)
        or {(r["id"], r["sample"]) for r in rows} != expected
    ):
        raise ValueError("Batch identity, settings or four-sample coverage is invalid.")
    if any(
        type(row["correct"]) is not bool
        or type(row["tokens"]) is not int
        or not 1 <= row["tokens"] <= 32768
        or row["finish_reason"] not in {"length", "stop"}
        or (row["finish_reason"] == "length" and row["tokens"] != 32768)
        for row in rows
    ):
        raise ValueError(
            "A batch has missing grades, aborted outputs or incomplete answer budgets."
        )
    if summarize(rows) != summary["metrics"]:
        raise ValueError("The batch metrics disagree with its per-sample records.")


def archived_batch(folder: Path) -> tuple[dict, list[dict]]:
    receipt = read_json(folder / "complete.json")
    if set(receipt["archives"]) != {"generations.jsonl.gz", "per_sample.jsonl.gz"}:
        raise ValueError("A completed batch must archive both answers and grades.")
    for filename, sha in receipt["archives"].items():
        if file_hash(folder / filename) != sha:
            raise ValueError(f"Batch archive checksum mismatch: {filename}")
    with gzip.open(folder / "per_sample.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    summary = read_json(folder / "summary.json")
    if file_hash(folder / "summary.json") != receipt["summary_sha256"]:
        raise ValueError("Batch summary changed after completion.")
    return summary, rows


def worker_state(path: Path, phase: str, **fields) -> None:
    process = psutil.Process()
    write_json(
        path,
        {
            "pid": process.pid,
            "created_at": process.create_time(),
            "phase": phase,
            **fields,
        },
    )


def worker(root: Path, manifest: dict, arm: str, gpu: int) -> None:
    from transformers import AutoTokenizer

    folder = root / "runs/evaluation" / manifest["name"]
    state = folder / f"{arm}-gpu{gpu}.json"
    worker_state(state, "verifying", arm=arm, gpu=gpu)
    if gpu not in manifest["profile"]["physical_gpus"]:
        raise ValueError("Worker GPU is outside the evaluation allocation.")
    model = manifest["models"][arm]
    if checkpoint_hashes(Path(model["path"])) != model["sha256"]:
        raise ValueError("The evaluated weights changed after selection.")
    if file_hash(Path(manifest["dataset"])) != manifest["dataset_sha256"]:
        raise ValueError("Benchmark inputs changed.")
    rows = {row["id"]: row for row in read_jsonl(Path(manifest["dataset"]))}
    engine = tokenizer = None
    for batch in manifest["batches"]:
        if batch["gpu"] != gpu:
            continue
        require_headroom(root, settings(root)["resources"], disk_bytes=1024**3)
        config = request_config(root, manifest, arm, batch)
        output = Path(config["output"])
        questions = [rows[identifier] for identifier in batch["ids"]]
        dataset = Path(config["dataset"])
        if dataset.exists() and read_jsonl(dataset) != questions:
            raise ValueError("Recorded batch questions differ from the manifest.")
        write_jsonl(dataset, questions)
        if (output / "complete.json").exists():
            saved, scores = archived_batch(output)
            validate_batch(saved, scores, config, batch["ids"], model["sha256"])
            continue
        if (output / "summary.json").exists():
            saved = read_json(output / "summary.json")
            scores = read_jsonl(output / "per_sample.jsonl")
        else:
            if engine is None:
                worker_state(state, "loading", arm=arm, gpu=gpu)
                engine = create_engine(config)
                tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
            worker_state(state, "generating", arm=arm, gpu=gpu, batch=batch["index"])
            immutable(output / "request.json", config)
            saved = evaluate(config, llm=engine, tokenizer=tokenizer)
            scores = read_jsonl(output / "per_sample.jsonl")
        validate_batch(saved, scores, config, batch["ids"], model["sha256"])
        archives = {}
        for name in ["generations.jsonl", "per_sample.jsonl"]:
            destination = output / (name + ".gz")
            compress_file(output / name, destination)
            archives[destination.name] = file_hash(destination)
        write_json(
            output / "complete.json",
            {
                "archives": archives,
                "summary_sha256": file_hash(output / "summary.json"),
            },
        )
        print(
            f"{arm} GPU{gpu}: saved batch {batch['index']:03d}, {len(scores)} samples", flush=True
        )
    worker_state(state, "completed", arm=arm, gpu=gpu)


def aggregate(root: Path, manifest: dict, arm: str) -> dict:
    records = []
    for batch in manifest["batches"]:
        config = request_config(root, manifest, arm, batch)
        summary, rows = archived_batch(Path(config["output"]))
        validate_batch(summary, rows, config, batch["ids"], manifest["models"][arm]["sha256"])
        records.extend(rows)
    if len(records) != 2000 or len({(row["id"], row["sample"]) for row in records}) != 2000:
        raise ValueError("The full arm is incomplete.")
    metrics = summarize(records)
    if metrics["problems"] != 500 or metrics["samples_per_problem"] != 4:
        raise ValueError("The full benchmark requires 500 questions with four samples each.")
    return {
        "complete": True,
        "metrics": metrics,
        "accuracy_percentage": 100 * metrics["accuracy"],
        "correct_samples": sum(row["correct"] for row in records),
        "correct_and_finished": sum(
            row["correct"] and row["finish_reason"] == "stop" for row in records
        ),
        "checkpoint_sha256": manifest["models"][arm]["sha256"],
    }


def run_campaign(root: Path, manifest: dict) -> None:
    folder = root / "runs/evaluation" / manifest["name"]
    arms = {}
    for arm in ARMS:
        missing = any(
            not (
                Path(request_config(root, manifest, arm, batch)["output"]) / "complete.json"
            ).exists()
            for batch in manifest["batches"]
        )
        if missing:
            require_gpu_headroom(
                manifest["profile"]["physical_gpus"],
                manifest["profile"]["gpu_memory_utilization"],
                manifest["profile"]["minimum_remaining_gpu_gib"],
            )
        write_json(folder / "progress.json", {"phase": "running", "arm": arm, "complete": False})
        children, descendants = [], {}
        try:
            for gpu in manifest["profile"]["physical_gpus"] if missing else []:
                with (folder / f"{arm}-gpu{gpu}.log").open("a") as log:
                    children.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "wsp.benchmark",
                                manifest["name"],
                                "--worker-arm",
                                arm,
                                "--gpu",
                                str(gpu),
                            ],
                            cwd=root,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    )
            while children:
                require_headroom(root, settings(root)["resources"])
                for child in children:
                    try:
                        for owned in psutil.Process(child.pid).children(recursive=True):
                            descendants[(owned.pid, owned.create_time())] = owned
                    except psutil.NoSuchProcess:
                        continue
                codes = [child.poll() for child in children]
                if any(code not in {None, 0} for code in codes):
                    raise RuntimeError(f"A {arm} worker failed; inspect per-GPU logs in {folder}.")
                if all(code == 0 for code in codes):
                    break
                time.sleep(3)
        finally:
            for child in children:
                if child.poll() is None:
                    stop_process_tree(child.pid)
                    child.wait()
            for owned in descendants.values():
                try:
                    alive = owned.is_running() and owned.status() not in {
                        psutil.STATUS_DEAD,
                        psutil.STATUS_ZOMBIE,
                    }
                except psutil.NoSuchProcess:
                    alive = False
                if alive:
                    stop_process_tree(owned.pid)
        arms[arm] = aggregate(root, manifest, arm)
        write_json(folder / f"{arm}.json", arms[arm])
    result = {
        "complete": True,
        "questions": 500,
        "samples_per_question": 4,
        "samples_per_arm": 2000,
        "metric": manifest["metric"],
        "dataset_sha256": manifest["dataset_sha256"],
        "models": arms,
        "accuracy_gain_percentage_points": (
            arms["edited"]["accuracy_percentage"] - arms["original"]["accuracy_percentage"]
        ),
    }
    write_json(folder / "result.json", result)
    write_json(folder / "progress.json", {"phase": "completed", "complete": True})
    print(json.dumps(result, indent=2), flush=True)


def status(root: Path, name: str) -> dict:
    folder = root / "runs/evaluation" / run_name(name)
    manifest = read_json(folder / "manifest.json")
    return {
        "name": name,
        "progress": read_json(folder / "progress.json")
        if (folder / "progress.json").exists()
        else {"phase": "prepared"},
        "completed_batches": {
            arm: len(list((folder / arm).glob("batch_*/complete.json"))) for arm in ARMS
        },
        "total_batches_per_model": len(manifest["batches"]),
        "result": str(folder / "result.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--edit", choices=published_edits(), default="run1")
    source.add_argument(
        "--run", help="A model-selected research final instead of a published edit."
    )
    parser.add_argument("--gpus", type=int, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    name = run_name(args.name)
    folder = ROOT / "runs/evaluation" / name
    if args.status:
        print(json.dumps(status(ROOT, name), indent=2))
        return
    if args.worker_arm:
        if args.gpu is None:
            parser.error("--worker-arm requires --gpu.")
        worker(ROOT, read_json(folder / "manifest.json"), args.worker_arm, args.gpu)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "campaign.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This evaluation already has a running controller.") from error
        manifest = prepare(ROOT, name, args.run, args.gpus, args.edit)
        if args.prepare_only:
            print("Prepared all 500 questions x four samples x two models.", flush=True)
            return

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            run_campaign(ROOT, manifest)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            write_json(
                folder / "progress.json",
                {
                    "phase": "failed",
                    "error": type(error).__name__,
                    "message": str(error),
                },
            )
            raise


if __name__ == "__main__":
    main()
