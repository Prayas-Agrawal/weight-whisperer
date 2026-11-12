import gzip
import hashlib
import json
import shutil
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_gzip(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n").encode()
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)


def compress_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as stream, temporary.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            shutil.copyfileobj(stream, compressed)
    temporary.replace(destination)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_hashes(model: Path) -> dict[str, str]:
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError("Cannot identify an observation checkpoint without weight shards.")
    return {path.name: file_hash(path) for path in shards}


def item_seed(identifier: str, seed: int) -> int:
    value = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(value[:4], "big") % (2**31)
