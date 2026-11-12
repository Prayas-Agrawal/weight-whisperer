"""Show where each published edit writes and which token direction it moves."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from wsp.config import ROOT, resolved_models
from wsp.io import read_json
from wsp.rebuild import published_edits, rebuild
from wsp.weights import TensorStore

EMBEDDING = "model.language_model.embed_tokens.weight"


def summarize_delta(
    delta: torch.Tensor, embedding: torch.Tensor, decode: Callable[[int], str], top: int = 3
) -> dict:
    left, values, _ = torch.linalg.svd(delta.double(), full_matrices=False)
    energy = values.square() / values.square().sum()
    summary = {
        "rank1_energy_percent": round(100 * float(energy[0]), 2),
        "rank2_energy_percent": round(100 * float(energy[:2].sum()), 2),
    }
    # Tied embedding rows are also the residual-stream directions read out as tokens.
    if delta.shape[0] == embedding.shape[1]:
        direction = left[:, 0].to(embedding.dtype)
        cosine = (embedding @ direction) / embedding.norm(dim=1).clamp_min(1e-12)
        summary["output_direction_tokens"] = [
            {
                "id": token,
                "token": decode(token),
                "abs_cosine": round(abs(float(cosine[token])), 3),
            }
            for token in cosine.abs().topk(top).indices.tolist()
        ]
    return summary


def describe(root: Path, edit: str) -> dict:
    from transformers import AutoTokenizer

    base = Path(resolved_models(root)["target"]["path"])
    original, edited = TensorStore(base), TensorStore(rebuild(root, edit=edit))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    embedding = original.load(EMBEDDING).float()
    tensors = []
    for row in read_json(root / "artifacts" / edit / "manifest.json")["tensors"]:
        before = original.load(row["name"]).float()
        delta = edited.load(row["name"]).float() - before
        tensors.append(
            {
                "name": row["name"],
                "shape": list(delta.shape),
                "changed_entries": row["entries"],
                "relative_change_percent": round(100 * float(delta.norm() / before.norm()), 2),
                **summarize_delta(delta, embedding, lambda token: tokenizer.decode([token])),
            }
        )
    return {"edit": edit, "tensors": tensors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", choices=published_edits(), action="append")
    args = parser.parse_args()
    torch.set_num_threads(8)
    for edit in args.edit or published_edits():
        print(json.dumps(describe(ROOT, edit), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
