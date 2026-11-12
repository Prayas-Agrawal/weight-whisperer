"""A per-investigation CPU Python namespace, retained across tool calls and quick cycles."""

import argparse
import ast
import contextlib
import json
import socket
import subprocess
import time
import traceback
import uuid
from pathlib import Path

from wsp.resources import stop_process_tree
from wsp.sandbox import audit_code, bootstrap_code, execution_command, limits


def execute_cell(code: str, namespace: dict) -> None:
    audit_code(code)
    tree = ast.parse(code)
    expression = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<tool-cell>", "exec"), namespace)
    if expression is not None:
        value = eval(compile(ast.Expression(expression.value), "<tool-cell>", "eval"), namespace)
        if value is not None:
            print(repr(value))


def serve(kernel_id: str) -> None:
    namespace = {"__name__": "__main__"}
    exec(bootstrap_code(), namespace)
    address = Path("/work/.python-kernel.sock")
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(address))
        address.chmod(0o600)
        server.listen(1)
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                raw = stream.readline(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Kernel request exceeds 2 MB.")
                request = json.loads(raw)
                if request["operation"] in {"ping", "stop"}:
                    result = {"kernel_id": kernel_id}
                elif request["operation"] == "execute":
                    turn = request["turn"]
                    if not isinstance(turn, int) or turn < 0:
                        raise ValueError("Invalid kernel turn.")
                    code = request["code"]
                    Path(f"/work/code_{turn:03d}.py").write_text(code)
                    output = Path(f"/work/python_{turn:03d}.log")
                    status = 0
                    with (
                        output.open("w") as log,
                        contextlib.redirect_stdout(log),
                        contextlib.redirect_stderr(log),
                    ):
                        try:
                            execute_cell(code, namespace)
                        except BaseException:
                            status = 1
                            traceback.print_exc()
                    with output.open() as log:
                        text = log.read(12001)
                    result = {
                        "kernel_id": kernel_id,
                        "exit_code": status,
                        "timed_out": False,
                        "output": text[:12000],
                        "output_truncated": len(text) > 12000,
                        "full_log": str(output),
                        "variables_persist": True,
                    }
                else:
                    raise ValueError("Unknown kernel operation.")
                stream.write((json.dumps(result) + "\n").encode())
                stream.flush()
                if request["operation"] == "stop":
                    break


def exchange(work: Path, request: dict, timeout: int) -> dict:
    address = str(work / ".python-kernel.sock")
    if len(address.encode()) >= 108:
        raise ValueError("Investigation path is too long for a local Unix socket.")
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(timeout)
        connection.connect(address)
        with connection.makefile("rwb") as stream:
            stream.write((json.dumps(request) + "\n").encode())
            stream.flush()
            reply = stream.readline(100_001)
            if not reply or len(reply) > 100_000:
                raise RuntimeError("Python kernel disconnected or returned an oversized response.")
            return json.loads(reply)


def owned_process(metadata: dict):
    import psutil

    try:
        process = psutil.Process(metadata["pid"])
        if process.create_time() != metadata["created_at"] or not process.is_running():
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def stop_kernel(work: Path) -> None:
    metadata_file = work.parent / "python-kernel.json"
    if not metadata_file.exists():
        return
    metadata = json.loads(metadata_file.read_text())
    process = owned_process(metadata)
    if process is not None:
        stop_process_tree(process.pid)
    (work / ".python-kernel.sock").unlink(missing_ok=True)


def ensure_kernel(
    root: Path,
    work: Path,
    model: Path,
    practice: Path,
) -> tuple[dict, bool]:
    import psutil

    metadata_file = work.parent / "python-kernel.json"
    expected = {"model": str(model), "practice": str(practice)}
    old = json.loads(metadata_file.read_text()) if metadata_file.exists() else None
    if old is not None and owned_process(old) is not None:
        if old["inputs"] != expected:
            raise ValueError("The live Python kernel belongs to different inputs.")
        if exchange(work, {"operation": "ping"}, 5)["kernel_id"] != old["kernel_id"]:
            raise RuntimeError("Python kernel identity mismatch.")
        return old, False
    work.mkdir(parents=True, exist_ok=True)
    (work / ".python-kernel.sock").unlink(missing_ok=True)
    identity = uuid.uuid4().hex
    arguments = ["-m", "wsp.python_kernel", "--serve", "--kernel-id", identity]
    command = execution_command(root, work, model, practice, arguments)
    with (work.parent / "python-kernel.log").open("a") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=limits,
        )
    metadata = {
        "pid": child.pid,
        "created_at": psutil.Process(child.pid).create_time(),
        "kernel_id": identity,
        "inputs": expected,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    deadline = time.monotonic() + 120
    while not (work / ".python-kernel.sock").exists():
        if child.poll() is not None:
            raise RuntimeError("CPU Python kernel failed to start; inspect python-kernel.log.")
        if time.monotonic() > deadline:
            stop_kernel(work)
            raise TimeoutError("CPU Python kernel startup timed out.")
        time.sleep(0.1)
    if exchange(work, {"operation": "ping"}, 5)["kernel_id"] != identity:
        raise RuntimeError("Started Python kernel has the wrong identity.")
    return metadata, old is not None


def execute_persistent(
    root: Path,
    work: Path,
    model: Path,
    practice: Path,
    code: str,
    turn: int,
    timeout: int = 1200,
) -> dict:
    audit_code(code)
    metadata, restarted = ensure_kernel(root, work, model, practice)
    try:
        result = exchange(work, {"operation": "execute", "code": code, "turn": turn}, timeout)
    except (TimeoutError, ConnectionError, RuntimeError) as error:
        stop_kernel(work)
        return {
            "exit_code": 124 if isinstance(error, TimeoutError) else 1,
            "timed_out": isinstance(error, TimeoutError),
            "error": type(error).__name__,
            "output": str(error),
            "kernel_id": metadata["kernel_id"],
            "variables_lost": True,
            "message": "CPU kernel stopped; variables were lost. Work files and edits remain.",
        }
    if result["kernel_id"] != metadata["kernel_id"]:
        raise RuntimeError("Response came from an unexpected CPU kernel.")
    result["kernel_restarted"] = restarted
    if restarted:
        result["restart_notice"] = "The prior CPU kernel exited; its variables were lost."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--kernel-id")
    parser.add_argument("--stop-work", type=Path)
    args = parser.parse_args()
    if args.serve and args.kernel_id:
        serve(args.kernel_id)
    elif args.stop_work:
        stop_kernel(args.stop_work.resolve())
    else:
        parser.error("Use --serve --kernel-id or --stop-work.")


if __name__ == "__main__":
    main()
