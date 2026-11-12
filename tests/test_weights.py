import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from wsp.sandbox import audit_code
from wsp.weights import (
    FROZEN_OUTPUT_TENSORS,
    TensorStore,
    WeightWorkspace,
    build_checkpoint,
    export_delta,
    export_workspace,
    load_delta,
    rebuild_delta,
    restore_workspace,
    tensor_hash,
    verify_checkpoint,
)

MATRIX = "model.language_model.layers.0.mlp.down_proj.weight"
EMBEDDING = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.norm.weight"


@pytest.fixture
def workspace(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {
            MATRIX: torch.arange(16).reshape(4, 4).bfloat16(),
            EMBEDDING: torch.ones(3, 4).bfloat16(),
            NORM: torch.ones(4).bfloat16(),
        },
        base / "model.safetensors",
    )
    return WeightWorkspace(base, tmp_path / "work")


def test_exact_reconstruction_preserves_every_other_tensor(workspace, tmp_path):
    values = workspace.load_original(MATRIX)
    values[0, 0] = 0.125
    workspace.save_edit(MATRIX, values)
    candidate = tmp_path / "candidate"
    report = build_checkpoint(workspace.model, workspace.edits, candidate)
    manifest = export_delta(workspace.model, candidate, tmp_path / "delta.npz")
    manifest["frozen_output_sha256"] = report["frozen_output_sha256"]
    rebuilt = tmp_path / "rebuilt"
    rebuild_delta(workspace.model, tmp_path / "delta.npz", manifest, rebuilt)
    assert verify_checkpoint(workspace.model, rebuilt, manifest)["changed_tensors"] == [MATRIX]
    assert torch.equal(TensorStore(rebuilt).load(MATRIX).float(), values)
    assert TensorStore(workspace.model).load(MATRIX)[0, 0] == 0
    with pytest.raises(ValueError, match="Wrong base"):
        rebuild_delta(candidate, tmp_path / "delta.npz", manifest, tmp_path / "wrong")


@pytest.mark.parametrize("name", [EMBEDDING, NORM])
def test_frozen_outputs_are_rejected_through_helpers_and_direct_files(workspace, tmp_path, name):
    values = workspace.load_original(name) * 2
    with pytest.raises(ValueError, match="frozen"):
        workspace.save_edit(name, values)
    assert workspace.edited() == []
    np.save(workspace.edits / f"{name}.npy", values.numpy())
    with pytest.raises(ValueError, match="frozen"):
        build_checkpoint(workspace.model, workspace.edits, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_internal_scaling_remains_available(workspace, tmp_path):
    values = workspace.load_original(MATRIX) * 2
    workspace.save_edit(MATRIX, values)
    result = build_checkpoint(workspace.model, workspace.edits, tmp_path / "scaled")
    assert result["frozen_output_sha256"] == {
        name: tensor_hash(workspace.store.load(name))
        for name in FROZEN_OUTPUT_TENSORS & set(workspace.names)
    }


def test_invalid_edits_and_inference_underflow_fail_explicitly(workspace, tmp_path):
    original = workspace.load_original(MATRIX)
    for bad in [original, original * float("nan"), torch.zeros(3)]:
        with pytest.raises(ValueError):
            workspace.save_edit(MATRIX, bad)
    values = original.clone()
    values[1, 1] += 0.0001
    workspace.save_edit(MATRIX, values)
    with pytest.raises(ValueError, match="vanishes"):
        build_checkpoint(workspace.model, workspace.edits, tmp_path / "underflow")
    assert not (tmp_path / "underflow").exists()


def test_workspace_snapshot_preserves_float32_values_before_rounding(workspace, tmp_path):
    values = workspace.load_original(MATRIX)
    values[1, 2] += 0.000123
    workspace.save_edit(MATRIX, values)
    path = tmp_path / "workspace.npz"
    manifest = export_workspace(workspace.model, workspace.edits.parent, path)
    restore_workspace(workspace.model, tmp_path / "recovered", path, manifest)
    restored = WeightWorkspace(workspace.model, tmp_path / "recovered").load(MATRIX)
    assert torch.equal(restored, values)
    assert not torch.equal(restored, values.bfloat16().float())
    with pytest.raises(FileExistsError, match="overwrite"):
        restore_workspace(workspace.model, tmp_path / "recovered", path, manifest)


def test_indexed_shard_names_are_supported(workspace, tmp_path):
    source = workspace.model / "model.safetensors"
    shard = workspace.model / "model.safetensors-00001-of-00001.safetensors"
    source.rename(shard)
    (workspace.model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in workspace.names}})
    )
    assert TensorStore(workspace.model).keys() == workspace.names


def test_actual_large_delta_is_chunked_and_rebuilt_exactly(workspace, tmp_path, monkeypatch):
    values = workspace.load_original(MATRIX)
    values[0, 0] = 0.125
    workspace.save_edit(MATRIX, values)
    build_checkpoint(workspace.model, workspace.edits, tmp_path / "candidate")

    def oversized(path, **arrays):
        np.savez(path, **arrays, padding=np.zeros(50_000_001, dtype=np.uint8))

    monkeypatch.setattr("wsp.weights.np.savez_compressed", oversized)
    output = tmp_path / "publish/delta.npz"
    manifest = export_delta(workspace.model, tmp_path / "candidate", output)
    index = json.loads(output.read_text())
    assert index["bytes"] > 50_000_000 and len(index["parts"]) == 2
    assert all(path.stat().st_size < 50_000_000 for path in output.parent.iterdir())
    rebuild_delta(workspace.model, output, manifest, tmp_path / "large-rebuilt")
    verify_checkpoint(workspace.model, tmp_path / "large-rebuilt", manifest)


def test_chunk_corruption_and_path_escape_are_rejected(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr("wsp.weights.DELTA_PART_BYTES", 128)
    values = workspace.load_original(MATRIX)
    values[0, 0] = 0.125
    workspace.save_edit(MATRIX, values)
    output = tmp_path / "delta.npz"
    export_workspace(workspace.model, workspace.edits.parent, output)
    index = json.loads(output.read_text())
    part = tmp_path / index["parts"][0]["name"]
    content = part.read_bytes()
    part.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="checksum"):
        with load_delta(output):
            pass
    index["parts"][0]["name"] = "../outside"
    output.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="part path"):
        with load_delta(output):
            pass


@pytest.mark.parametrize(
    "code",
    [
        "loss.backward()",
        "import torch.optim as optim",
        "x.requires_grad_(True)",
        "torch.set_grad_enabled(flag)",
        "scipy.optimize.minimize(loss)",
    ],
)
def test_training_calls_are_detected(code):
    with pytest.raises(ValueError):
        audit_code(code)


def test_closed_form_analysis_and_disabling_gradients_are_available():
    audit_code("u, s, v = torch.linalg.svd(load(NAMES[0]))")
    audit_code("torch.set_grad_enabled(False)\nx.requires_grad_(False)")
