"""Verify the real runtime: CPU sandbox by default; with --full, also GPU tools and the editor."""

import argparse
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from wsp.action_transport import (
    PROTOCOL_TOKENS,
    THINK_END,
    framed_response_format,
    parse_json_action,
)
from wsp.config import ROOT, resolved_models
from wsp.io import read_json, read_jsonl
from wsp.python_kernel import execute_persistent, stop_kernel


def check_kernel(root: Path, model: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="runtime-check-", dir=root / "runs") as temporary:
        work = Path(temporary) / "work"
        work.mkdir()
        code = (
            "import socket\n"
            "assert not torch.cuda.is_available()\n"
            "assert not Path('/root').exists()\n"
            "assert not Path('/proc/1').exists()\n"
            "assert socket.if_nameindex() == [(1, 'lo')]\n"
            "name = 'model.language_model.layers.0.mlp.down_proj.weight'\n"
            "weights = load_original(name)\n"
            "assert tuple(weights.shape) == (1024, 3584)\n"
            "weights[0, 0] += 0.125\n"
            "save_edit(name, weights)\n"
            "drop_edit(name)\n"
            "for frozen in [\n"
            "    'model.language_model.embed_tokens.weight', 'model.language_model.norm.weight'\n"
            "]:\n"
            "    try:\n"
            "        save_edit(frozen, load_original(frozen) * 2)\n"
            "    except ValueError as error:\n"
            "        assert 'frozen' in str(error)\n"
            "    else:\n"
            "        raise AssertionError('Frozen tensor accepted')\n"
            "assert edited() == []\n"
            "counter = 41\n"
            "counter\n"
        )
        try:
            first = execute_persistent(root, work, model, root / "runs/data/practice", code, 1)
            if first["exit_code"] != 0 or first["output"].splitlines()[-1:] != ["41"]:
                raise RuntimeError(f"Real CPU tool check failed: {first}")
            second = execute_persistent(
                root,
                work,
                model,
                root / "runs/data/practice",
                "assert counter == 41\ncounter += 1\ncounter",
                2,
            )
            if (
                second["exit_code"] != 0
                or second["output"].splitlines()[-1:] != ["42"]
                or second["kernel_id"] != first["kernel_id"]
                or second["kernel_restarted"]
            ):
                raise RuntimeError(f"Persistent tool state check failed: {second}")
        finally:
            stop_kernel(work)
    return {
        "real_weight_read_edit_verified": True,
        "frozen_output_rejection_verified": True,
        "cpu_network_and_filesystem_isolation_verified": True,
        "variables_persist_across_calls": True,
        "first_call_output": first["output"].strip(),
    }


def check_framing(editor: dict) -> dict:
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm.entrypoints.generate.base.protocol import validate_structural_tag_response_format

    tokenizer = AutoTokenizer.from_pretrained(editor["path"], local_files_only=True)
    model = read_json(Path(editor["path"]) / "config.json")
    vocab_size = model.get("text_config", model)["vocab_size"]
    stops = read_json(Path(editor["path"]) / "generation_config.json")["eos_token_id"]
    stops = [stops] if isinstance(stops, int) else stops
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=stops
    )
    specification = framed_response_format(tokenizer)
    validate_structural_tag_response_format(specification)
    grammar = xgr.GrammarCompiler(info, max_threads=4).compile_structural_tag(specification)
    matcher = xgr.GrammarMatcher(grammar)
    mask = xgr.allocate_token_bitmask(1, vocab_size)

    def available(token):
        matcher.fill_next_token_bitmask(mask)
        return bool(int(mask[0, token // 32]) & (1 << (token % 32)))

    def accept(tokens):
        for token in tokens:
            if not matcher.accept_token(token):
                raise RuntimeError(f"Grammar rejected expected token {token}.")

    if any(available(token) for token in stops):
        raise RuntimeError("The grammar permits termination before an action.")
    reasoning = tokenizer.encode(
        "I can inspect the data, consider alternatives, and check my calculations.\n" * 1024,
        add_special_tokens=False,
    )
    accept(reasoning)
    if any(available(token) for token in stops):
        raise RuntimeError("The free-reasoning region permits premature termination.")
    accept([tokenizer.convert_tokens_to_ids(THINK_END)])
    payload = (
        json.dumps(
            {
                "action": "forward",
                "body": {
                    "type": "teacher_forced",
                    "model": "original",
                    "texts": [" ".join(PROTOCOL_TOKENS)],
                },
            }
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    accept(tokenizer.encode(payload, add_special_tokens=False))
    if not all(available(token) for token in stops):
        raise RuntimeError("A complete action cannot terminate normally.")
    accept([stops[0]])
    if not matcher.is_terminated():
        raise RuntimeError("The completed action did not terminate.")
    return {
        "actual_editor_tokenizer_and_http_schema_verified": True,
        "free_reasoning_tokens_tested": len(reasoning),
        "complete_forward_object_required": True,
    }


def temporary_investigation(root: Path):
    from wsp.editor import Experiment

    name = "runtime-check-" + uuid.uuid4().hex[:12]
    return Experiment(root, name), root / "runs/research" / name


def check_target_tools(root: Path) -> dict:
    experiment, folder = temporary_investigation(root)
    try:
        generated = experiment.forward(
            {
                "type": "generate",
                "model": "original",
                "prompts": ["What is 2 + 3? Reply with the number only."],
                "n": 1,
                "max_tokens": 32768,
            }
        )
        rows = read_jsonl(experiment.work / "gen_1_original.jsonl")
        if generated["max_tokens"] != 32768 or len(rows) != 1 or rows[0]["tokens"] < 1:
            raise RuntimeError(f"Real target generation failed: {generated}")
        observed = experiment.forward(
            {
                "type": "teacher_forced",
                "model": "original",
                "texts": ["1 + 1 = 2"],
                "return": {"hidden_states": [24], "token_logprobs": True, "topk": 5},
            }
        )
        outputs = experiment.work / "fwd_2"
        if not (outputs / "hidden_24.npy").is_file() or not (outputs / "meta.json").is_file():
            raise RuntimeError(f"Real teacher-forced observation failed: {observed}")
    finally:
        shutil.rmtree(folder)
    return {
        "target_generation_tokens": rows[0]["tokens"],
        "target_generation_finish_reason": rows[0]["finish_reason"],
        "teacher_forced_arrays_verified": True,
    }


def check_editor_turn(root: Path) -> dict:
    from wsp import autonomy

    experiment, folder = temporary_investigation(root)
    try:
        experiment.run(max_steps=1)
        record = read_json(folder / "turns/0001.json")
        if record.get("malformed_reply_excluded_from_history") or "tool_result" not in record:
            raise RuntimeError(f"The live editor action did not complete: {record}")
        action, _ = parse_json_action(record["content"])
        autonomy.audit(root, folder, experiment.messages, experiment.turn)
    finally:
        stop_kernel(folder / "work")
        shutil.rmtree(folder)
    return {
        "live_editor_action": action,
        "editor_completion_tokens": record["usage"]["completion_tokens"],
        "reasoning_preserved": bool(record["reasoning_content"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also test editor framing, real target tools on GPU and one live editor action.",
    )
    args = parser.parse_args()
    models = resolved_models()
    report = {"python_tool": check_kernel(ROOT, Path(models["target"]["path"]))}
    if args.full:
        if "editor" not in models:
            raise ValueError("Prepare both models before the full runtime check.")
        report["action_framing"] = check_framing(models["editor"])
        report["target_tools"] = check_target_tools(ROOT)
        report["editor"] = check_editor_turn(ROOT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
