"""Portable paths and the single released experiment configuration."""

import re
from pathlib import Path

from wsp.io import read_json

ROOT = Path(__file__).resolve().parents[2]


def run_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError("Use an alphanumeric run name with optional hyphens or underscores.")
    return value


def settings(root: Path = ROOT) -> dict:
    return read_json(root / "configs/experiment.json")


def resolved_models(root: Path = ROOT) -> dict:
    path = root / "runs/models.json"
    if not path.is_file():
        raise FileNotFoundError("Run 'bash scripts/run.sh prepare' before using model weights.")
    models = read_json(path)
    pinned = read_json(root / "configs/models.json")
    for role, value in models.items():
        if role not in pinned or any(
            value[key] != pinned[role][key] for key in ["model_id", "revision"]
        ):
            raise ValueError(f"The resolved {role} model differs from its pinned identity.")
        if not Path(value["path"]).is_dir():
            raise FileNotFoundError(f"Cached model is unavailable: {role}; rerun prepare.")
    return models


def initial_messages(root: Path = ROOT) -> list[dict]:
    return [
        {"role": role, "content": (root / f"configs/{role}_prompt.txt").read_text()}
        for role in ["system", "user"]
    ]
