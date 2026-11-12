"""Start or reuse the pinned local editor; stop only a process started by this checkout."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from wsp.config import ROOT, resolved_models, settings
from wsp.io import read_json, write_json
from wsp.resources import require_gpu_headroom, require_headroom, stop_process_tree


def health(editor: dict) -> bool:
    try:
        response = httpx.get(f"http://127.0.0.1:{editor['port']}/v1/models", timeout=5)
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False
    response.raise_for_status()
    if not any(
        row["id"] == "wsp-editor" and row.get("root") == editor["path"]
        for row in response.json()["data"]
    ):
        raise RuntimeError("The editor endpoint belongs to a different model.")
    return True


def ensure_editor(root: Path = ROOT) -> None:
    editor = resolved_models(root).get("editor")
    if editor is None:
        raise FileNotFoundError("Run prepare --models target editor before research.")
    if health(editor):
        print("Reusing the pinned editor; no restart.", flush=True)
        return
    require_headroom(root, settings(root)["resources"], ram_bytes=16 * 1024**3)
    require_gpu_headroom(editor["physical_gpus"], editor["gpu_memory_utilization"])
    folder = root / "runs"
    folder.mkdir(exist_ok=True)
    with (folder / "editor.log").open("a") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "wsp.services", "serve"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        folder / "editor_process.json",
        {
            "pid": child.pid,
            "created_at": psutil.Process(child.pid).create_time(),
            "model": editor["path"],
            "port": editor["port"],
        },
    )
    deadline = time.monotonic() + 1800
    try:
        while not health(editor):
            if child.poll() is not None:
                raise RuntimeError(f"Editor startup failed; inspect {folder / 'editor.log'}.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Editor did not become ready within 30 minutes.")
            require_headroom(root, settings(root)["resources"])
            time.sleep(3)
    except BaseException:
        if child.poll() is None:
            stop_process_tree(child.pid)
            child.wait()
        raise
    print("Editor ready; it stays loaded until stop-editor.", flush=True)


def stop_editor(root: Path = ROOT) -> None:
    path = root / "runs/editor_process.json"
    if not path.exists():
        raise RuntimeError("This checkout has no owned editor process to stop.")
    record = read_json(path)
    try:
        process = psutil.Process(record["pid"])
    except psutil.NoSuchProcess:
        print("The recorded editor process is already gone.", flush=True)
        return
    if process.create_time() != record["created_at"]:
        raise RuntimeError("The recorded PID belongs to a different process; refusing to stop it.")
    command = process.cmdline()
    if "vllm.entrypoints.openai.api_server" not in command:
        raise RuntimeError("The recorded PID is not the expected editor API server.")
    for flag, value in {
        "--model": record["model"],
        "--served-model-name": "wsp-editor",
        "--host": "127.0.0.1",
        "--port": str(record["port"]),
    }.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise RuntimeError(f"Editor identity changed at {flag}; refusing to stop it.")
    stop_process_tree(process.pid)
    print("Stopped only this checkout's editor and its workers.", flush=True)


def serve(root: Path = ROOT) -> None:
    editor = resolved_models(root)["editor"]
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", editor["port"]))
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, editor["physical_gpus"])),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
    )
    import json

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        editor["path"],
        "--served-model-name",
        "wsp-editor",
        "--host",
        "127.0.0.1",
        "--port",
        str(editor["port"]),
        "--tensor-parallel-size",
        str(editor["tensor_parallel_size"]),
        "--gpu-memory-utilization",
        str(editor["gpu_memory_utilization"]),
        "--max-model-len",
        str(editor["max_model_len"]),
        "--max-num-seqs",
        str(editor["max_num_seqs"]),
        "--max-num-batched-tokens",
        "8192",
        "--reasoning-parser",
        editor["reasoning_parser"],
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--hf-overrides",
        json.dumps(editor["hf_overrides"]),
        "--structured-outputs-config",
        json.dumps(editor["structured_outputs_config"]),
    ]
    if editor["enforce_eager"]:
        command.append("--enforce-eager")
    os.execv(sys.executable, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "start", "stop"])
    args = parser.parse_args()
    {"serve": serve, "start": ensure_editor, "stop": stop_editor}[args.action]()


if __name__ == "__main__":
    main()
