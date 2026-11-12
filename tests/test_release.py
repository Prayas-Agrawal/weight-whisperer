import gzip
import json
import shutil

import pytest

from wsp.config import ROOT, run_name
from wsp.verify import verify_result


def test_published_scores_and_artifacts_reproduce_both_runs():
    assert verify_result() == {
        "verified": True,
        "questions": 500,
        "samples_per_model": 2000,
        "original_accuracy_percent": 7.7,
        "runs": {
            "run1": {"accuracy_percent": 50.4, "gain_percentage_points": 42.7},
            "run2": {"accuracy_percent": 51.55, "gain_percentage_points": 43.85},
        },
    }


@pytest.mark.parametrize("scores", ["original", "run1", "run2"])
def test_score_corruption_is_detected_before_reporting_metrics(tmp_path, scores):
    for directory in ["results", "artifacts", "configs"]:
        shutil.copytree(ROOT / directory, tmp_path / directory)
    path = tmp_path / f"results/scores/{scores}.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        lines = list(stream)
    changed = json.loads(lines[0])
    changed["correct"] = not changed["correct"]
    lines[0] = json.dumps(changed) + "\n"
    path.write_bytes(gzip.compress("".join(lines).encode(), mtime=0))
    with pytest.raises(ValueError, match="checksum"):
        verify_result(tmp_path)


def test_altered_weight_delta_is_rejected(tmp_path):
    for directory in ["results", "artifacts", "configs"]:
        shutil.copytree(ROOT / directory, tmp_path / directory)
    with (tmp_path / "artifacts/run2/edit.npz").open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ValueError, match="run2 delta"):
        verify_result(tmp_path)


@pytest.mark.parametrize("name", ["../private", "/tmp/file", "", "two words", ".hidden"])
def test_names_cannot_escape_the_run_directory(name):
    with pytest.raises(ValueError):
        run_name(name)
