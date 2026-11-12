"""Download pinned weights and datasets; do not download another experiment's edited weights."""

import argparse
import urllib.request
from pathlib import Path

from wsp.config import ROOT, settings
from wsp.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from wsp.resources import require_headroom


def prepare_models(root: Path, roles: list[str], cache: Path) -> None:
    from huggingface_hub import snapshot_download

    pinned = read_json(root / "configs/models.json")
    if not roles or any(role not in pinned for role in roles):
        raise ValueError("Choose target, editor, or both model roles.")
    cache.mkdir(parents=True, exist_ok=True)
    output = root / "runs/models.json"
    resolved = read_json(output) if output.exists() else {}
    for role in roles:
        info = pinned[role]
        snapshot = (
            cache
            / ("models--" + info["model_id"].replace("/", "--"))
            / "snapshots"
            / info["revision"]
        )
        present = sum(
            path.stat().st_size for path in snapshot.glob("*.safetensors") if path.is_file()
        )
        require_headroom(
            root,
            settings(root)["resources"],
            disk_bytes=max(0, info["weight_bytes"] - present),
        )
        path = snapshot_download(
            repo_id=info["model_id"],
            revision=info["revision"],
            cache_dir=cache,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "README.md",
                "LICENSE",
            ],
            max_workers=8,
        )
        resolved[role] = {**info, "path": str(Path(path).resolve())}
        write_json(output, resolved)
        print(f"Prepared {role}: {info['model_id']} @ {info['revision']}", flush=True)


def prepare_practice(root: Path) -> None:
    config = read_json(root / "configs/data.json")["practice"]
    output = root / "runs/data/practice"
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for name, info in config["files"].items():
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("Practice filenames must be direct JSONL children.")
        path = output / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(info["url"], timeout=120) as response:
                    data = response.read(50 * 1024**2 + 1)
                if len(data) > 50 * 1024**2:
                    raise ValueError(f"Practice download is unexpectedly large: {name}")
                temporary.write_bytes(data)
                if file_hash(temporary) != info["sha256"]:
                    raise ValueError(f"Pinned practice file checksum mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Practice file changed: {name}")
        rows = read_jsonl(path)
        if len(rows) != info["rows"]:
            raise ValueError(f"Practice row count mismatch: {name}")
        for row in rows:
            if not all(key in row for key in ["id", "source", "problem", "answer"]):
                raise ValueError(f"Missing practice fields in {name}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate practice ID: {row['id']}")
            seen.add(row["id"])
        print(f"Verified practice {name}: {len(rows)} questions", flush=True)


def prepare_evaluation(root: Path) -> None:
    from datasets import load_dataset

    source = read_json(root / "configs/data.json")["evaluation"]
    output = root / "runs/data/evaluation/math500.jsonl"
    expected_hash = read_json(root / "results/math500.json")["benchmark"]["dataset_sha256"]
    if output.exists():
        if file_hash(output) != expected_hash:
            raise ValueError("The benchmark file differs from the published dataset hash.")
    else:
        dataset = load_dataset(
            source["dataset"],
            split=source["split"],
            revision=source["revision"],
            cache_dir=str(root / "data"),
        )
        rows = [
            {"id": f"math500/test/{i}", "problem": row["problem"], "answer": row["answer"]}
            for i, row in enumerate(dataset)
        ]
        if len(rows) != 500:
            raise ValueError("Expected the complete 500-question benchmark.")
        write_jsonl(output, rows)
        if file_hash(output) != expected_hash:
            raise ValueError("Downloaded benchmark differs from the published dataset hash.")
    print("Verified MATH-500: all 500 pinned questions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["target", "editor"], default=["target"])
    parser.add_argument("--cache", type=Path, default=ROOT / "hf-cache")
    args = parser.parse_args()
    prepare_models(ROOT, args.models, args.cache.resolve())
    prepare_practice(ROOT)
    prepare_evaluation(ROOT)


if __name__ == "__main__":
    main()
