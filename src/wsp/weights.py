import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

DELTA_PART_BYTES = 48_000_000
FROZEN_OUTPUT_TENSORS = frozenset(
    {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.lm_head.weight",
        "model.language_model.lm_head.bias",
        "lm_head.weight",
        "lm_head.bias",
    }
)


def validate_edit_target(name: str) -> None:
    if name in FROZEN_OUTPUT_TENSORS:
        raise ValueError(f"Tensor is frozen by this run's edit policy: {name}")


def delta_index(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        if stream.read(2) == b"PK":
            return None
    index = json.loads(path.read_text())
    if not isinstance(index, dict) or index.get("format") != "wsp-chunked-npz-v1":
        raise ValueError("Unknown delta container format.")
    if not isinstance(index.get("parts"), list) or not index["parts"]:
        raise ValueError("Chunked delta has no parts.")
    return index


def delta_part_path(path: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.startswith(path.name + ".part-")
    ):
        raise ValueError("Invalid delta part path.")
    part = path.parent / name
    if part.is_symlink() or part.resolve().parent != path.parent.resolve():
        raise ValueError("Delta parts must remain inside their artifact directory.")
    return part


def store_delta(source: Path, output: Path) -> None:
    previous = delta_index(output)
    old_parts = (
        [delta_part_path(output, part["name"]) for part in previous["parts"]] if previous else []
    )
    new_parts = []
    if source.stat().st_size <= DELTA_PART_BYTES:
        source.replace(output)
    else:
        parts, digest, total = [], hashlib.sha256(), 0
        with source.open("rb") as stream:
            while data := stream.read(DELTA_PART_BYTES):
                sha = hashlib.sha256(data).hexdigest()
                name = f"{output.name}.part-{len(parts):04d}-{sha[:16]}"
                temporary = source.parent / name
                temporary.write_bytes(data)
                destination = delta_part_path(output, name)
                temporary.replace(destination)
                new_parts.append(destination)
                parts.append({"name": name, "bytes": len(data), "sha256": sha})
                digest.update(data)
                total += len(data)
        index = {
            "format": "wsp-chunked-npz-v1",
            "bytes": total,
            "sha256": digest.hexdigest(),
            "parts": parts,
        }
        temporary = source.parent / "index.json"
        temporary.write_text(json.dumps(index, indent=2) + "\n")
        temporary.replace(output)
    for path in old_parts:
        if path not in new_parts:
            path.unlink()


@contextmanager
def load_delta(path: Path):
    index = delta_index(path)
    if index is None:
        with np.load(path, allow_pickle=False) as data:
            yield data
        return
    digest, total = hashlib.sha256(), 0
    with tempfile.TemporaryFile() as joined:
        for part in index["parts"]:
            filename = delta_part_path(path, part["name"])
            if not 0 < part["bytes"] <= 50_000_000 or filename.stat().st_size != part["bytes"]:
                raise ValueError("Delta part size mismatch.")
            part_digest = hashlib.sha256()
            with filename.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    joined.write(block)
                    part_digest.update(block)
                    digest.update(block)
                    total += len(block)
            if part_digest.hexdigest() != part["sha256"]:
                raise ValueError("Delta part checksum mismatch.")
        if total != index["bytes"] or digest.hexdigest() != index["sha256"]:
            raise ValueError("Complete delta checksum mismatch.")
        joined.seek(0)
        with np.load(joined, allow_pickle=False) as data:
            yield data


class TensorStore:
    def __init__(self, model: Path):
        self.model = model
        index = model / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            self.files = {}
            for path in sorted(model.glob("*.safetensors")):
                with safe_open(path, framework="pt") as tensors:
                    for name in tensors.keys():
                        if name in self.files:
                            raise ValueError(f"Duplicate checkpoint tensor: {name}")
                        self.files[name] = path.name
        if not self.files:
            raise ValueError(f"No safetensors weights found in {model}")
        for filename in set(self.files.values()):
            if Path(filename).name != filename or not (model / filename).is_file():
                raise ValueError(f"Invalid or missing checkpoint shard: {filename}")

    def keys(self) -> list[str]:
        return sorted(self.files)

    def load(self, name: str) -> torch.Tensor:
        with safe_open(self.model / self.files[name], framework="pt") as tensors:
            return tensors.get_tensor(name)


class WeightWorkspace:
    def __init__(self, model: Path, work: Path):
        self.model = model
        self.edits = work / "edited"
        self.edits.mkdir(parents=True, exist_ok=True)
        self.store = TensorStore(model)
        self.names = self.store.keys()

    def load_original(self, name: str) -> torch.Tensor:
        if name not in self.names:
            raise KeyError(f"Unknown tensor: {name}")
        return self.store.load(name).float()

    def load(self, name: str) -> torch.Tensor:
        original = self.load_original(name)
        path = self.edits / f"{name}.npy"
        return torch.from_numpy(np.load(path, allow_pickle=False)) if path.exists() else original

    def save_edit(self, name: str, tensor: torch.Tensor | np.ndarray) -> dict:
        original = self.load_original(name)
        validate_edit_target(name)
        values = (
            tensor.detach().cpu().float()
            if isinstance(tensor, torch.Tensor)
            else torch.from_numpy(np.asarray(tensor, dtype=np.float32))
        )
        if values.shape != original.shape or not torch.isfinite(values).all():
            raise ValueError(
                "An edit must preserve the tensor shape and contain only finite values."
            )
        if torch.equal(original, values):
            raise ValueError(
                "This is an unchanged tensor, not an edit; use drop_edit to restore it."
            )
        np.save(self.edits / f"{name}.npy", values.numpy(), allow_pickle=False)
        return {"name": name, "changed_float32_entries": int((values != original).sum())}

    def drop_edit(self, name: str) -> None:
        if name not in self.names:
            raise KeyError(f"Unknown tensor: {name}")
        (self.edits / f"{name}.npy").unlink(missing_ok=True)

    def edited(self) -> list[str]:
        return sorted(path.name.removesuffix(".npy") for path in self.edits.glob("*.npy"))


def build_checkpoint(
    base: Path,
    edits: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen checkpoint: {output}")
    source = TensorStore(base)
    tensors = {name: source.load(name) for name in source.keys()}
    changes = []
    for path in sorted(edits.glob("*.npy")):
        name = path.name.removesuffix(".npy")
        if name not in tensors:
            raise ValueError(f"Forbidden or unknown edited tensor: {name}")
        validate_edit_target(name)
        old = tensors[name]
        proposal = np.load(path, allow_pickle=False)
        if proposal.shape != tuple(old.shape) or not np.isfinite(proposal).all():
            raise ValueError(f"Invalid edit: {name}")
        # Revalidate sandbox-written files outside the sandbox before evaluation.
        original = old.float()
        values = torch.from_numpy(proposal).float()
        new = values.to(old.dtype)
        if not torch.isfinite(new).all():
            raise ValueError(f"Edit overflows inference dtype: {name}")
        count = int((old != new).sum())
        if count == 0:
            raise ValueError(f"Edit vanishes when cast to inference dtype: {name}")
        changes.append(
            {
                "name": name,
                "shape": list(old.shape),
                "changed_entries": count,
                "dtype": str(old.dtype),
                "sha256": hashlib.sha256(new.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "relative_l2_change": float((new.float() - original).norm() / original.norm()),
            }
        )
        tensors[name] = new.contiguous()
    if not changes:
        raise ValueError("No nonzero edits were supplied.")
    changed_entries = sum(row["changed_entries"] for row in changes)
    save_checkpoint(base, tensors, output)
    report = {"tensors": changes, "changed_entries": changed_entries}
    report["frozen_output_sha256"] = {
        name: tensor_hash(tensors[name]) for name in sorted(FROZEN_OUTPUT_TENSORS & tensors.keys())
    }
    return report


def save_checkpoint(base: Path, tensors: dict[str, torch.Tensor], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        folder = Path(temporary)
        for path in base.iterdir():
            if (
                path.is_file()
                and path.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken"}
                and path.name != "model.safetensors.index.json"
            ):
                shutil.copyfile(path, folder / path.name)
        save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})
        folder.rename(output)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def export_delta(base: Path, edited: Path, output: Path) -> dict:
    original, candidate = TensorStore(base), TensorStore(edited)
    if original.keys() != candidate.keys():
        raise ValueError("A delta must preserve every original tensor name.")
    return export_tensor_deltas(
        ((name, original.load(name), candidate.load(name)) for name in original.keys()), output
    )


def export_tensor_deltas(
    pairs: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    output: Path,
    allow_empty: bool = False,
) -> dict:
    arrays, manifest = {}, []
    for name, before, after in pairs:
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Delta changes tensor metadata: {name}")
        if not torch.isfinite(after).all():
            raise ValueError(f"Nonfinite delta values: {name}")
        indices = torch.nonzero(before.flatten() != after.flatten()).flatten()
        if not len(indices):
            continue
        key = f"tensor_{len(manifest)}"
        values = after.contiguous().view(torch.uint8).reshape(-1, after.element_size())[indices]
        arrays[f"{key}_indices"] = indices.numpy()
        arrays[f"{key}_bytes"] = values.numpy()
        manifest.append(
            {
                "key": key,
                "name": name,
                "shape": list(before.shape),
                "dtype": str(before.dtype),
                "entries": len(indices),
                "base_sha256": tensor_hash(before),
                "edited_sha256": tensor_hash(after),
            }
        )
    if not manifest and not allow_empty:
        raise ValueError("No changed entries to export.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delta-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "delta.npz"
        np.savez_compressed(candidate, **arrays)
        store_delta(candidate, output)
    return {"format": "wsp-sparse-raw-bytes-v1", "tensors": manifest}


def export_workspace(base: Path, work: Path, output: Path) -> dict:
    paths = sorted((work / "edited").glob("*.npy"))
    if not paths:
        manifest = export_tensor_deltas([], output, allow_empty=True)
        return {**manifest, "workspace_precision": "float32"}
    source = TensorStore(base)
    pairs = (
        (
            path.name.removesuffix(".npy"),
            source.load(path.name.removesuffix(".npy")).float(),
            torch.from_numpy(np.load(path, allow_pickle=False)),
        )
        for path in paths
    )
    manifest = export_tensor_deltas(pairs, output, allow_empty=True)
    return {**manifest, "workspace_precision": "float32"}


def apply_tensor_delta(tensor: torch.Tensor, row: dict, data) -> torch.Tensor:
    tensor = tensor.clone().contiguous()
    if tensor_hash(tensor) != row["base_sha256"]:
        raise ValueError(f"Wrong base tensor for {row['name']}")
    indices = data[f"{row['key']}_indices"]
    values = data[f"{row['key']}_bytes"]
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= tensor.numel())
        or values.shape != (len(indices), tensor.element_size())
        or values.dtype != np.uint8
    ):
        raise ValueError(f"Malformed delta for {row['name']}")
    tensor.view(torch.uint8).reshape(-1, tensor.element_size())[indices] = torch.from_numpy(values)
    if tensor_hash(tensor) != row["edited_sha256"]:
        raise ValueError(f"Rebuilt tensor hash mismatch for {row['name']}")
    return tensor


def restore_workspace(
    base: Path,
    work: Path,
    delta: Path,
    manifest: dict,
) -> None:
    if manifest["format"] != "wsp-sparse-raw-bytes-v1":
        raise ValueError("Unknown delta format.")
    workspace = WeightWorkspace(base, work)
    for row in manifest["tensors"]:
        validate_edit_target(row["name"])
    if workspace.edited():
        raise FileExistsError("Refusing to overwrite an existing weight workspace.")
    float32 = manifest.get("workspace_precision") == "float32"
    with load_delta(delta) as data:
        for row in manifest["tensors"]:
            original = workspace.store.load(row["name"])
            if float32:
                original = original.float()
            restored = apply_tensor_delta(original, row, data)
            workspace.save_edit(row["name"], restored)


def rebuild_delta(
    base: Path,
    delta: Path,
    manifest: dict,
    output: Path,
) -> None:
    if manifest["format"] != "wsp-sparse-raw-bytes-v1":
        raise ValueError("Unknown delta format.")
    for row in manifest["tensors"]:
        validate_edit_target(row["name"])
    source = TensorStore(base)
    tensors = {name: source.load(name) for name in source.keys()}
    with load_delta(delta) as data:
        for row in manifest["tensors"]:
            tensors[row["name"]] = apply_tensor_delta(tensors[row["name"]], row, data)
    save_checkpoint(base, tensors, output)


def verify_checkpoint(base: Path, candidate: Path, manifest: dict) -> dict:
    original, edited = TensorStore(base), TensorStore(candidate)
    if original.keys() != edited.keys():
        raise ValueError("The reconstructed checkpoint changes tensor names.")
    expected = {row["name"]: row for row in manifest["tensors"]}
    if len(expected) != len(manifest["tensors"]):
        raise ValueError("Duplicate tensors in the edit manifest.")
    for name in expected:
        validate_edit_target(name)
    protected = {}
    for name in original.keys():
        before, after = original.load(name), edited.load(name)
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Tensor metadata changed: {name}")
        if name in expected:
            row = expected[name]
            if (
                tensor_hash(before) != row["base_sha256"]
                or tensor_hash(after) != row["edited_sha256"]
            ):
                raise ValueError(f"Reconstructed tensor hash mismatch: {name}")
        elif not torch.equal(before, after):
            raise ValueError(f"Unrecorded tensor modification: {name}")
        if name in FROZEN_OUTPUT_TENSORS:
            protected[name] = tensor_hash(after)
    if "frozen_output_sha256" in manifest and protected != manifest["frozen_output_sha256"]:
        raise ValueError("Protected output hashes differ from the published manifest.")
    return {"changed_tensors": sorted(expected), "frozen_output_sha256": protected}
