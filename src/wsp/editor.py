"""A single autonomous weight-editing workflow with a fixed prompt and full-budget tools."""

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import torch

from wsp import autonomy
from wsp.action_transport import (
    decode_framed_response,
    framed_response_format,
    parse_json_action,
)
from wsp.config import ROOT, initial_messages, resolved_models, run_name, settings
from wsp.editor_settings import build_request, transport_messages
from wsp.evaluate import require_full_reasoning_budget
from wsp.grading import paired_comparison
from wsp.io import read_json, read_jsonl, write_json, write_json_gzip, write_jsonl
from wsp.observation_spec import normalize_capture_options, observation_layout
from wsp.python_kernel import execute_persistent, stop_kernel
from wsp.resources import require_headroom, wait_guarded
from wsp.services import ensure_editor
from wsp.weights import build_checkpoint, export_delta, export_workspace


def generation_sampling(config: dict, request: dict, count: int) -> dict:
    sampling = dict(config["sampling"])
    samples = request.get("n", sampling["samples_per_problem"])
    if type(samples) is not int or samples < 1:
        raise ValueError("n must be a positive integer.")
    total = count * samples * (2 if request["model"] == "both" else 1)
    if total > config["tools"]["max_practice_generations_per_call"]:
        raise ValueError("Requested generations exceed the per-call capacity; use a smaller batch.")
    sampling.update(samples_per_problem=samples, max_tokens=request.get("max_tokens", 32768))
    require_full_reasoning_budget(sampling)
    return sampling


def practice_selection(request: dict, practice: dict) -> list[dict]:
    if not isinstance(request.get("model"), str) or request["model"] not in {
        "original",
        "edited",
        "both",
    }:
        raise ValueError("Generation model must be original, edited or both.")
    ids, prompts = request.get("problems", []), request.get("prompts", [])
    for values in [ids, prompts]:
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("Supply distinct nonempty problem IDs or custom messages.")
    if not ids and not prompts:
        raise ValueError("Supply at least one practice problem or custom message.")
    if any(identifier not in practice for identifier in ids):
        raise ValueError("Only IDs from the mounted practice data are permitted.")
    return [practice[identifier] for identifier in ids] + [
        {
            "id": "custom/" + autonomy.digest(prompt),
            "problem": prompt,
            "answer": None,
            "source": "custom",
            "custom_prompt": True,
        }
        for prompt in prompts
    ]


def capture_inputs(request: dict, work: Path) -> dict:
    fields = [key for key in ["texts", "chats", "contexts_file"] if key in request]
    if len(fields) != 1:
        raise ValueError("Supply exactly one of texts, chats or contexts_file.")
    field = fields[0]
    if field != "contexts_file":
        return {field: request[field]}
    if not isinstance(request[field], str) or not request[field]:
        raise ValueError("contexts_file must be a nonempty /work path.")
    path = Path(request[field])
    if not path.is_relative_to("/work"):
        raise ValueError("contexts_file must be under /work.")
    resolved = (work / path.relative_to("/work")).resolve()
    if not resolved.is_relative_to(work.resolve()):
        raise ValueError("contexts_file escapes /work.")
    values = read_json(resolved)
    if not isinstance(values, list) or not values:
        raise ValueError("Context input must be a nonempty list.")
    if all(isinstance(value, str) for value in values):
        return {"texts": values}
    if all(isinstance(value, list) for value in values):
        return {"chats": values}
    raise ValueError("Do not mix raw texts and chat message lists.")


class Experiment:
    def __init__(self, root: Path, name: str):
        torch.set_num_threads(8)
        self.root, self.name = root, run_name(name)
        self.config = settings(root)
        self.models = resolved_models(root)
        self.base = Path(self.models["target"]["path"])
        self.folder = root / "runs/research" / self.name
        self.work = self.folder / "work"
        self.practice_dir = root / "runs/data/practice"
        self.practice = {}
        for filename in read_json(root / "configs/data.json")["practice"]["files"]:
            for row in read_jsonl(self.practice_dir / filename):
                if row["id"] in self.practice:
                    raise ValueError(f"Duplicate practice ID: {row['id']}")
                self.practice[row["id"]] = row
        if (self.folder / "final.json").exists():
            raise ValueError("This investigation is finalized; do not overwrite or refine it.")
        if self.folder.exists():
            state = read_json(self.folder / "state.json")
            for key in ["turn", "python_calls", "forward_calls", "forward_attempts", "messages"]:
                setattr(self, key, state[key])
            if (self.folder / "turns" / f"{self.turn + 1:04d}.json").exists():
                raise RuntimeError(
                    "An interrupted proposal exists; its evidence must not be overwritten."
                )
            autonomy.audit(root, self.folder, self.messages, self.turn)
        else:
            self.work.mkdir(parents=True)
            self.turn = self.python_calls = self.forward_calls = self.forward_attempts = 0
            self.messages = initial_messages(root)
            autonomy.seal(root, self.folder)
            self.save_state()

    def save_state(self) -> None:
        autonomy.audit(self.root, self.folder, self.messages, self.turn)
        recovery = self.folder / "recovery"
        manifest = export_workspace(self.base, self.work, recovery / "workspace.npz")
        write_json(recovery / "workspace_manifest.json", manifest)
        value = {
            key: getattr(self, key)
            for key in ["turn", "python_calls", "forward_calls", "forward_attempts", "messages"]
        }
        write_json(self.folder / "state.json", value)
        write_json_gzip(recovery / "state.json.gz", value)
        write_json(
            self.folder / "progress.json", {key: value[key] for key in value if key != "messages"}
        )

    def worker(self, module: str, config: dict, label: str) -> None:
        request = self.folder / "requests" / f"{label}.json"
        require_headroom(self.root, self.config["resources"], ram_bytes=4 * 1024**3)
        write_json(request, config)
        with request.with_suffix(".log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", module, "--config", str(request)],
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status = wait_guarded(
                child,
                self.root,
                self.config["resources"],
                timeout=self.config["tools"]["forward_timeout_seconds"],
            )
        if status:
            raise subprocess.CalledProcessError(status, child.args)

    def forward(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Forward input must be a JSON object.")
        number = self.forward_calls + 1
        self.forward_attempts += 1
        label = f"fwd_{number}_attempt_{self.forward_attempts}"
        checkpoints = self.folder / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=label + "-", dir=checkpoints) as temporary:
            candidate = None

            def selected_model(role):
                nonlocal candidate
                if role == "original":
                    return self.base
                if role != "edited":
                    raise ValueError("Unknown model selection.")
                if candidate is None:
                    candidate = Path(temporary) / "model"
                    report = build_checkpoint(self.base, self.work / "edited", candidate)
                    write_json(candidate / "edit_manifest.json", report)
                    artifact = self.folder / "candidates" / label
                    manifest = export_delta(self.base, candidate, artifact / "delta.npz")
                    manifest["frozen_output_sha256"] = report["frozen_output_sha256"]
                    write_json(artifact / "manifest.json", manifest)
                return candidate

            if request.get("type") == "generate":
                rows = practice_selection(request, self.practice)
                sampling = generation_sampling(self.config, request, len(rows))
                activations = request.get("activations")
                if activations is not None:
                    if (
                        not isinstance(activations, dict)
                        or set(activations) - {"layers", "mlp"}
                        or type(activations.get("mlp", False)) is not bool
                    ):
                        raise ValueError("activations requires layers and an optional boolean mlp.")
                    layers = activations.get("layers", [])
                    if (
                        not isinstance(layers, list)
                        or any(type(layer) is not int or not 0 <= layer <= 24 for layer in layers)
                        or len(layers) != len(set(layers))
                    ):
                        raise ValueError(
                            "Activation layers must be distinct integers from 0 to 24."
                        )
                dataset = self.folder / "requests" / f"gen_{number}_questions.jsonl"
                write_jsonl(dataset, rows)
                roles = ["original", "edited"] if request["model"] == "both" else [request["model"]]
                results, samples = {}, {}
                for role in roles:
                    output = self.folder / "forwards" / f"gen_{number}_{role}"
                    config = {
                        "model": str(selected_model(role)),
                        "dataset": str(dataset),
                        "output": str(output),
                        "physical_gpu": self.models["target"]["physical_gpus"][0],
                        **self.config["research_engine"],
                        "sampling": sampling,
                        "activations": activations,
                    }
                    self.worker("wsp.evaluate", config, f"gen_{number}_{role}")
                    shutil.copyfile(
                        output / "generations.jsonl", self.work / f"gen_{number}_{role}.jsonl"
                    )
                    summary = read_json(output / "summary.json")
                    write_json(self.work / f"gen_{number}_{role}_meta.json", summary)
                    results[role] = summary["metrics"]
                    samples[role] = read_jsonl(output / "per_sample.jsonl")
                    if activations is not None:
                        shutil.copytree(
                            output / "activations", self.work / f"gen_{number}_{role}_acts"
                        )
                if len(roles) == 2:
                    results["paired"] = paired_comparison(samples["original"], samples["edited"])
                result = {
                    "type": "generate",
                    "number": number,
                    "results": results,
                    "max_tokens": 32768,
                    "files": [f"/work/gen_{number}_{role}.jsonl" for role in roles],
                    "metadata_files": [f"/work/gen_{number}_{role}_meta.json" for role in roles],
                    "activation_folders": (
                        [f"/work/gen_{number}_{role}_acts" for role in roles] if activations else []
                    ),
                }
            elif request.get("type") == "teacher_forced":
                from transformers import AutoTokenizer

                from wsp.capture import encode_inputs

                config = {
                    **capture_inputs(request, self.work),
                    **normalize_capture_options(request.get("return", {}), "tokens"),
                    "enable_thinking": request.get("enable_thinking", True),
                    "add_generation_prompt": request.get("add_generation_prompt", False),
                    "max_texts": self.config["tools"]["capture_max_texts"],
                    "max_input_tokens": self.config["tools"]["capture_max_input_tokens"],
                    "max_total_tokens": self.config["tools"]["capture_max_total_tokens"],
                    "max_output_bytes": 6_000_000_000,
                    "physical_gpu": self.models["target"]["physical_gpus"][0],
                    "output": str(self.work / f"fwd_{number}"),
                }
                tokenizer = AutoTokenizer.from_pretrained(self.base, local_files_only=True)
                lengths = list(map(len, encode_inputs(config, tokenizer)))
                plan = observation_layout(
                    lengths, config, read_json(self.base / "config.json")["text_config"]
                )
                require_headroom(
                    self.root, self.config["resources"], disk_bytes=plan["estimated_bytes"]
                )
                config["model"] = str(selected_model(request.get("model")))
                self.worker("wsp.capture", config, f"teacher_{number}")
                result = {
                    "type": "teacher_forced",
                    "number": number,
                    "folder": f"/work/fwd_{number}",
                    "metadata": read_json(self.work / f"fwd_{number}/meta.json"),
                }
            else:
                raise ValueError("Unknown forward type.")
        self.forward_calls += 1
        return result

    def finish(self, body: str, record: dict) -> None:
        report = {
            "turns": record["turn"],
            "python_calls": self.python_calls,
            "forward_calls": self.forward_calls,
            "final_response": body,
            "status": "no_edit",
            "selection": "model_final",
        }
        if list((self.work / "edited").glob("*.npy")):
            candidate = self.folder / "checkpoints/final"
            report["edit"] = build_checkpoint(self.base, self.work / "edited", candidate)
            manifest = export_delta(self.base, candidate, self.folder / "final_delta.npz")
            manifest["frozen_output_sha256"] = report["edit"]["frozen_output_sha256"]
            write_json(self.folder / "delta_manifest.json", manifest)
            report["status"] = "frozen"
        receipt = autonomy.finalize(self.root, self.folder, self.messages, report)
        write_json(self.folder / "final.json", report)
        write_json(self.folder / "final_receipt.json", receipt)
        write_json(
            self.folder / "progress.json",
            {
                "phase": "completed",
                "model_selected_final": True,
                "turn": record["turn"],
                "status": report["status"],
            },
        )
        stop_kernel(self.work)

    def run(self, max_steps: int | None = None) -> None:
        from transformers import AutoTokenizer

        ensure_editor(self.root)
        editor = self.models["editor"]
        tokenizer = AutoTokenizer.from_pretrained(editor["path"], local_files_only=True)
        controls = list(set(tokenizer.all_special_tokens) | set(tokenizer.get_added_vocab()))
        failures = steps = 0
        with httpx.Client(base_url=f"http://127.0.0.1:{editor['port']}", timeout=14400) as client:
            while max_steps is None or steps < max_steps:
                autonomy.audit(self.root, self.folder, self.messages, self.turn)
                if self.turn and self.turn % self.config["handoff_turns"] == 0:
                    messages = autonomy.handoff(self.root, self.folder, self.messages, self.turn)
                    if messages is not None:
                        self.messages = messages
                        self.save_state()
                require_headroom(self.root, self.config["resources"])
                messages = transport_messages(self.messages, controls)
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    **editor["generation"]["chat_template_kwargs"],
                )
                request = build_request(messages, editor, len(tokens))
                request.update(
                    response_format=framed_response_format(tokenizer), return_token_ids=True
                )
                response = client.post("/v1/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
                next_turn = self.turn + 1
                write_json_gzip(self.folder / "responses" / f"{next_turn:04d}.json.gz", payload)
                choice = payload["choices"][0]
                record = {
                    "turn": next_turn,
                    "content": choice["message"].get("content") or "",
                    "usage": payload["usage"],
                    "finish_reason": choice["finish_reason"],
                    "request_settings": {
                        key: value for key, value in request.items() if key != "messages"
                    },
                }
                path = self.folder / "turns" / f"{next_turn:04d}.json"
                write_json(path, record)
                decoded = parsed = False
                try:
                    reasoning, content = decode_framed_response(choice.get("token_ids"), tokenizer)
                    record.update(content=content, reasoning_content=reasoning)
                    self.messages.append(autonomy.assistant_message(record))
                    decoded = True
                    action, body = parse_json_action(content)
                    parsed = True
                    failures = 0
                    if action == "python":
                        self.python_calls += 1
                        result = execute_persistent(
                            self.root,
                            self.work,
                            self.base,
                            self.practice_dir,
                            body,
                            next_turn,
                            timeout=self.config["tools"]["python_timeout_seconds"],
                        )
                    elif action == "forward":
                        result = self.forward(json.loads(body))
                    else:
                        write_json(path, record)
                        self.finish(body, record)
                        return
                except (ValueError, KeyError, SyntaxError) as error:
                    result = {"error": type(error).__name__, "message": str(error)}
                    if not decoded:
                        record["malformed_reply_excluded_from_history"] = True
                    if not parsed:
                        failures += 1
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    record["tool_failure"] = str(error)
                    write_json(path, record)
                    raise
                record["tool_result"] = result
                write_json(path, record)
                self.messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(result) + autonomy.TOOL_CONTINUATION,
                    }
                )
                self.turn = next_turn
                self.save_state()
                steps += 1
                print(
                    json.dumps(
                        {
                            "turn": self.turn,
                            "python_calls": self.python_calls,
                            "forward_calls": self.forward_calls,
                        }
                    ),
                    flush=True,
                )
                if failures >= 2:
                    raise RuntimeError(
                        "Two incomplete model responses; stopped without substituting a candidate."
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument(
        "--steps", type=int, help="Pause after completed actions; do not shorten answers."
    )
    args = parser.parse_args()
    name = run_name(args.name)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive.")
    locks = ROOT / "runs/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"research-{name}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This investigation already has a running controller.") from error
        Experiment(ROOT, name).run(args.steps)


if __name__ == "__main__":
    main()
