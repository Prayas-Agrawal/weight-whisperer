"""Seal research inputs and audit model-authored handoffs and final selection."""

import copy
import hashlib
import json
from pathlib import Path

from wsp.action_transport import parse_json_action
from wsp.config import initial_messages, resolved_models
from wsp.io import checkpoint_hashes, file_hash, read_json, write_json, write_json_gzip

TOOL_CONTINUATION = "\nContinue toward a working result, or submit FINAL."


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def inputs(root: Path) -> dict:
    files = [
        *sorted((root / "configs").glob("*")),
        *sorted((root / "src/wsp").glob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "scripts/run.sh",
        root / "runs/models.json",
        *sorted((root / "runs/data/practice").glob("*.jsonl")),
    ]
    return {str(path.relative_to(root)): file_hash(path) for path in files if path.is_file()}


def shard_identity(base: Path) -> dict:
    return {
        path.name: {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(base.glob("*.safetensors"))
    }


def seal(root: Path, folder: Path) -> None:
    if any((folder / "work").iterdir()) or (folder / "protocol.json").exists():
        raise RuntimeError("Research must start with an empty workspace and a new protocol.")
    base = Path(resolved_models(root)["target"]["path"])
    value = {
        "version": 1,
        "initial_messages": initial_messages(root),
        "locked_inputs": inputs(root),
        "original_checkpoint_sha256": checkpoint_hashes(base),
        "original_shard_identity": shard_identity(base),
        "tool_continuation": TOOL_CONTINUATION,
        "controller_research_guidance_messages": 0,
        "initial_state": "original_weights_empty_workspace",
        "selection": "model_final_only",
    }
    write_json(folder / "protocol.json", {**value, "sha256": digest(value)})


def verify_seal(root: Path, folder: Path, runtime: bool = True) -> dict:
    value = read_json(folder / "protocol.json")
    if digest({key: item for key, item in value.items() if key != "sha256"}) != value["sha256"]:
        raise RuntimeError("The research protocol changed.")
    if runtime:
        if inputs(root) != value["locked_inputs"]:
            raise RuntimeError(
                "The sealed prompt, data or runtime changed; do not continue this run."
            )
        if (
            shard_identity(Path(resolved_models(root)["target"]["path"]))
            != value["original_shard_identity"]
        ):
            raise RuntimeError("The original checkpoint changed during research.")
    return value


def assistant_message(record: dict) -> dict:
    return {
        "role": "assistant",
        "content": record["content"],
        "reasoning_content": record["reasoning_content"],
    }


def handoff_content(notes: str, edited: list[str]) -> str:
    return (
        "Automatic state handoff. The original task, tools, budgets and operational notices "
        "are unchanged. Your workspace and Python variables remain available. No research "
        "summary or new hypothesis was supplied by the controller.\n"
        "Your currently saved edited tensors: "
        + json.dumps(edited)
        + "\nYour own notes from /work/research_notes.txt, verbatim:\n\n"
        + notes
    )


def recorded_messages(folder: Path, turn: int) -> list[dict]:
    protocol = read_json(folder / "protocol.json")
    messages = copy.deepcopy(protocol["initial_messages"])
    for index in range(1, turn + 1):
        record = read_json(folder / "turns" / f"{index:04d}.json")
        if record["turn"] != index or "tool_result" not in record:
            raise RuntimeError(f"Missing completed tool record for turn {index}.")
        if not record.get("malformed_reply_excluded_from_history"):
            messages.append(assistant_message(record))
        messages.append(
            {
                "role": "user",
                "content": json.dumps(record["tool_result"]) + protocol["tool_continuation"],
            }
        )
        handoff = folder / "handoffs" / f"{index:04d}.json"
        if handoff.exists():
            saved = read_json(handoff)
            if (
                saved["before_transcript_sha256"] != digest(messages)
                or saved["model_notes_sha256"] != digest(saved["model_notes"])
                or saved["content"]
                != handoff_content(saved["model_notes"], saved["edited_tensors"])
            ):
                raise RuntimeError("Model-note handoff history does not verify.")
            messages = copy.deepcopy(protocol["initial_messages"])
            messages.append({"role": "user", "content": saved["content"]})
    return messages


def audit(root: Path, folder: Path, messages: list[dict], turn: int) -> None:
    verify_seal(root, folder)
    if messages != recorded_messages(folder, turn):
        raise RuntimeError("Unrecorded instructions or altered model/tool history detected.")


def handoff(root: Path, folder: Path, messages: list[dict], turn: int) -> list[dict] | None:
    audit(root, folder, messages, turn)
    destination = folder / "handoffs" / f"{turn:04d}.json"
    note = folder / "work/research_notes.txt"
    if destination.exists() or not note.exists():
        return None
    if note.is_symlink() or not note.is_file() or note.stat().st_size > 256000:
        raise ValueError("The model's handoff notes must be a regular text file below 256KB.")
    notes = note.read_text()
    if not notes.strip():
        return None
    edited = sorted(path.stem for path in (folder / "work/edited").glob("*.npy"))
    write_json_gzip(folder / "handoffs" / f"{turn:04d}_history.json.gz", messages)
    write_json(
        destination,
        {
            "turn": turn,
            "model_notes": notes,
            "model_notes_sha256": digest(notes),
            "edited_tensors": edited,
            "content": handoff_content(notes, edited),
            "before_transcript_sha256": digest(messages),
        },
    )
    return recorded_messages(folder, turn)


def finalize(root: Path, folder: Path, messages: list[dict], report: dict) -> dict:
    turn = report["turns"]
    audit(root, folder, messages[:-1], turn - 1)
    record = read_json(folder / "turns" / f"{turn:04d}.json")
    action, body = parse_json_action(record["content"])
    if (
        action != "final"
        or messages[-1] != assistant_message(record)
        or report["final_response"] != body
    ):
        raise RuntimeError("Only the editor's recorded final action may select a candidate.")
    return {
        "model_selected_final": True,
        "final_turn": turn,
        "final_record_sha256": file_hash(folder / "turns" / f"{turn:04d}.json"),
        "transcript_sha256": digest(messages),
        "final_report_sha256": digest(report),
        "controller_research_guidance_messages": 0,
        "own_note_handoffs": len(list((folder / "handoffs").glob("[0-9][0-9][0-9][0-9].json"))),
    }


def require_final(root: Path, folder: Path) -> dict:
    verify_seal(root, folder, runtime=False)
    if not (folder / "final.json").exists() or not (folder / "final_receipt.json").exists():
        raise RuntimeError("Evaluation is blocked until the model submits its final candidate.")
    final = read_json(folder / "final.json")
    receipt = read_json(folder / "final_receipt.json")
    record = read_json(folder / "turns" / f"{receipt['final_turn']:04d}.json")
    messages = recorded_messages(folder, receipt["final_turn"] - 1) + [assistant_message(record)]
    action, body = parse_json_action(record["content"])
    if (
        not receipt["model_selected_final"]
        or action != "final"
        or final["final_response"] != body
        or receipt["final_report_sha256"] != digest(final)
        or receipt["transcript_sha256"] != digest(messages)
        or receipt["final_record_sha256"]
        != file_hash(folder / "turns" / f"{receipt['final_turn']:04d}.json")
    ):
        raise RuntimeError("The model-selected final does not match its audit receipt.")
    if final["status"] != "frozen":
        raise RuntimeError("The editor finalized without an edit; no performance gain is claimed.")
    return final
