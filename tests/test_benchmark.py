import copy
from pathlib import Path

import pytest

from wsp import benchmark
from wsp.config import settings
from wsp.grading import summarize
from wsp.io import checkpoint_hashes, file_hash, read_json, read_jsonl, write_json, write_jsonl


def questions():
    return [
        {"id": f"math500/test/{i}", "problem": f"Question {i}", "answer": "1"} for i in range(500)
    ]


def test_all_500_questions_and_four_samples_are_planned_on_chosen_gpus():
    plan = benchmark.plan(questions(), settings()["evaluation"])
    assert len(plan) == 21
    assert {row["gpu"] for row in plan} == {0, 1, 2, 3}
    assert sum(len(row["ids"]) * 4 for row in plan) == 2000
    assert [identifier for row in plan for identifier in row["ids"]] == [
        row["id"] for row in questions()
    ]
    assert len(plan[-1]["ids"]) == 20


@pytest.mark.parametrize("change", ["subset", "duplicates", "samples", "concurrency", "gpus"])
def test_weakened_or_incomplete_plans_are_refused(change):
    rows, profile = questions(), copy.deepcopy(settings()["evaluation"])
    if change == "subset":
        rows.pop()
    elif change == "duplicates":
        rows[-1] = rows[0]
    elif change == "samples":
        profile["samples_per_problem"] = 1
    elif change == "concurrency":
        profile["problems_per_batch"] = 100
    else:
        profile["physical_gpus"] = [0, 0]
    with pytest.raises(ValueError):
        benchmark.plan(rows, profile)


def scores(ids):
    return [
        {
            "id": identifier,
            "sample": sample,
            "tokens": 100 if sample == 0 else 32768,
            "correct": sample == 0,
            "finish_reason": "stop" if sample == 0 else "length",
            "thinking_closed": sample == 0,
            "grading_status": "graded",
        }
        for identifier in ids
        for sample in range(4)
    ]


@pytest.fixture
def worker_run(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixed model identity")
    profile = copy.deepcopy(settings()["evaluation"])
    profile["physical_gpus"] = [0]
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, questions())
    manifest = {
        "name": "full",
        "profile": profile,
        "dataset": str(dataset),
        "dataset_sha256": file_hash(dataset),
        "batches": benchmark.plan(questions(), profile),
        "sampling": {**settings()["sampling"], "samples_per_problem": 4},
        "models": {
            arm: {"path": str(model), "sha256": checkpoint_hashes(model)} for arm in benchmark.ARMS
        },
    }
    calls = {"engine": 0, "evaluate": 0}
    monkeypatch.setattr(benchmark, "require_headroom", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "settings", lambda root: {"resources": {}})
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object())

    def create(config):
        calls["engine"] += 1
        return object()

    def evaluate(config, llm, tokenizer):
        calls["evaluate"] += 1
        output = Path(config["output"])
        rows = scores([row["id"] for row in read_jsonl(Path(config["dataset"]))])
        summary = {
            "config": config,
            "capability_measurement": True,
            "model_checkpoint_sha256": manifest["models"]["original"]["sha256"],
            "dataset_sha256": file_hash(Path(config["dataset"])),
            "metrics": summarize(rows),
        }
        write_jsonl(output / "per_sample.jsonl", rows)
        write_jsonl(
            output / "generations.jsonl", [{**row, "text": "recorded answer"} for row in rows]
        )
        write_json(output / "summary.json", summary)
        return summary

    monkeypatch.setattr(benchmark, "create_engine", create)
    monkeypatch.setattr(benchmark, "evaluate", evaluate)
    return manifest, calls


def test_full_worker_reuses_engine_and_resumes_verified_batches(tmp_path, worker_run):
    manifest, calls = worker_run
    benchmark.worker(tmp_path, manifest, "original", 0)
    assert calls == {"engine": 1, "evaluate": 21}
    benchmark.worker(tmp_path, manifest, "original", 0)
    assert calls == {"engine": 1, "evaluate": 21}
    result = benchmark.aggregate(tmp_path, manifest, "original")
    assert result["complete"] is True
    assert result["metrics"]["problems"] == 500 and result["metrics"]["samples"] == 2000
    assert result["correct_samples"] == 500 and result["accuracy_percentage"] == 25.0
    assert result["correct_and_finished"] == 500


def test_corrupt_batch_is_not_silently_regenerated(tmp_path, worker_run):
    manifest, calls = worker_run
    benchmark.worker(tmp_path, manifest, "original", 0)
    archive = tmp_path / "runs/evaluation/full/original/batch_000/generations.jsonl.gz"
    archive.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.worker(tmp_path, manifest, "original", 0)
    assert calls["evaluate"] == 21


def test_manifest_cannot_silently_change(tmp_path):
    path = tmp_path / "manifest.json"
    benchmark.immutable(path, {"samples": 4})
    with pytest.raises(ValueError, match="settings changed"):
        benchmark.immutable(path, {"samples": 1})


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing",
        "short_length",
        "not_boolean",
        "aborted",
        "settings",
        "model",
        "dataset",
        "metrics",
    ],
)
def test_batch_validation_rejects_incomplete_or_changed_measurements(tmp_path, change):
    dataset = tmp_path / "questions.jsonl"
    write_jsonl(dataset, questions()[:1])
    config = {
        "sampling": {**settings()["sampling"], "samples_per_problem": 4},
        "dataset": str(dataset),
    }
    ids, hashes = ["math500/test/0"], {"model.safetensors": "original"}
    rows = scores(ids)
    summary = {
        "config": copy.deepcopy(config),
        "capability_measurement": True,
        "model_checkpoint_sha256": hashes,
        "dataset_sha256": file_hash(dataset),
        "metrics": summarize(rows),
    }
    benchmark.validate_batch(summary, rows, config, ids, hashes)
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "short_length":
        rows[1]["tokens"] = 8192
    elif change == "not_boolean":
        rows[0]["correct"] = 1
    elif change == "aborted":
        rows[0]["finish_reason"] = "abort"
    elif change == "settings":
        summary["config"]["sampling"]["temperature"] = 0.1
    elif change == "model":
        summary["model_checkpoint_sha256"] = {}
    elif change == "dataset":
        write_jsonl(dataset, questions()[1:2])
    else:
        summary["metrics"]["accuracy"] = 1.0
    with pytest.raises(ValueError):
        benchmark.validate_batch(summary, rows, config, ids, hashes)


def test_partial_evaluation_cannot_be_aggregated(tmp_path, worker_run):
    manifest, _ = worker_run
    with pytest.raises(FileNotFoundError):
        benchmark.aggregate(tmp_path, manifest, "original")


def test_worker_failure_stops_only_owned_running_helpers(tmp_path, monkeypatch):
    stopped, children = [], []
    folder = tmp_path / "runs/evaluation/failure"
    profile = copy.deepcopy(settings()["evaluation"])
    profile["physical_gpus"] = [0, 1]
    manifest = {
        "name": "failure",
        "profile": profile,
        "batches": [{"index": 0, "gpu": 0}],
        "models": {"original": {"path": "unused"}},
        "sampling": settings()["sampling"],
    }

    class Child:
        def __init__(self, *args, **kwargs):
            self.pid = 100000000 + len(children)
            self.code = None if not children else 1
            children.append(self)

        def poll(self):
            return self.code

        def wait(self):
            self.code = 0
            return self.code

    def missing_process(pid):
        raise benchmark.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(benchmark.subprocess, "Popen", Child)
    monkeypatch.setattr(benchmark.psutil, "Process", missing_process)
    monkeypatch.setattr(benchmark, "settings", lambda root: {"resources": {}})
    monkeypatch.setattr(benchmark, "require_headroom", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "require_gpu_headroom", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "stop_process_tree", stopped.append)
    with pytest.raises(RuntimeError, match="worker failed"):
        benchmark.run_campaign(tmp_path, manifest)
    assert stopped == [children[0].pid]
    assert read_json(folder / "progress.json")["complete"] is False
    assert not (folder / "result.json").exists()
