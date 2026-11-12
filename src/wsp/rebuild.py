"""Reconstruct a published edit, or an audited model-selected final."""

import argparse
from pathlib import Path

import torch

from wsp.config import ROOT, resolved_models, run_name, settings
from wsp.io import read_json, write_json
from wsp.resources import require_headroom
from wsp.weights import rebuild_delta, verify_checkpoint


def published_edits(root: Path = ROOT) -> list[str]:
    return sorted(read_json(root / "results/math500.json")["runs"])


def rebuild(root: Path, name: str | None = None, edit: str = "run1") -> Path:
    if name is None:
        if edit not in published_edits(root):
            raise ValueError(f"Choose a published edit: {', '.join(published_edits(root))}.")
        source = root / "artifacts" / edit
        delta, manifest_path = source / "edit.npz", source / "manifest.json"
        output = root / "runs/published" / edit
    else:
        from wsp.autonomy import require_final

        source = root / "runs/research" / run_name(name)
        require_final(root, source)
        delta, manifest_path = source / "final_delta.npz", source / "delta_manifest.json"
        output = source / "checkpoints/final"
    base = Path(resolved_models(root)["target"]["path"])
    manifest = read_json(manifest_path)
    require_headroom(
        root, settings(root)["resources"], disk_bytes=3 * 1024**3, ram_bytes=4 * 1024**3
    )
    torch.set_num_threads(8)
    if not output.exists():
        rebuild_delta(base, delta, manifest, output)
    report = verify_checkpoint(base, output, manifest)
    write_json(output / "verification.json", report)
    print(f"Verified exact reconstructed edit: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--edit", choices=published_edits(), default="run1")
    source.add_argument("--run", help="A self-finalized research run instead of a published edit.")
    args = parser.parse_args()
    rebuild(ROOT, args.run, args.edit)


if __name__ == "__main__":
    main()
