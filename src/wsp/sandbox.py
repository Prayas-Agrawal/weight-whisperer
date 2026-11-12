"""Network-isolated, CPU-only execution of model-written analysis code."""

import ast
import resource
from pathlib import Path


def audit_code(code: str) -> None:
    tree = ast.parse(code)
    disabling = set()
    keywords = {"set_grad_enabled": "mode", "requires_grad_": "requires_grad"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in keywords:
            continue
        arguments = node.args
        if not arguments and len(node.keywords) == 1:
            keyword = node.keywords[0]
            if keyword.arg == keywords[node.func.attr]:
                arguments = [keyword.value]
        elif node.keywords:
            continue
        if len(arguments) == 1 and isinstance(arguments[0], ast.Constant):
            if arguments[0].value is False:
                disabling.add(node.func)
    prohibited = {
        "backward",
        "requires_grad_",
        "autograd",
        "optim",
        "enable_grad",
        "set_grad_enabled",
        "fit",
        "fit_transform",
        "minimize",
        "least_squares",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in prohibited and node not in disabling:
            raise ValueError(f"Training/gradient operation forbidden: {node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(part in prohibited for part in node.module.split(".")):
                raise ValueError(f"Training/gradient import forbidden: {node.module}")
        if isinstance(node, ast.Import):
            for name in node.names:
                if any(part in prohibited for part in name.name.split(".")):
                    raise ValueError(f"Training/gradient import forbidden: {name.name}")


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (96 * 1024**3, 96 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**3, 4 * 1024**3))


def bootstrap_code() -> str:
    return (
        "from pathlib import Path\n"
        "import json, numpy as np, torch\n"
        "from transformers import AutoTokenizer\n"
        "from wsp.weights import WeightWorkspace\n"
        "torch.set_num_threads(4)\n"
        "torch.set_grad_enabled(False)\n"
        "workspace = WeightWorkspace(Path('/weights'), Path('/work'))\n"
        "NAMES = workspace.names\n"
        "load = workspace.load\n"
        "load_original = workspace.load_original\n"
        "save_edit = workspace.save_edit\n"
        "drop_edit = workspace.drop_edit\n"
        "edited = workspace.edited\n"
        "tok = AutoTokenizer.from_pretrained('/weights', local_files_only=True)\n"
    )


def execution_command(
    root: Path, work: Path, model: Path, practice: Path, arguments: list[str]
) -> list[str]:
    command = [
        "bwrap",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--dir",
        "/proc",
        "--ro-bind",
        "/proc/cpuinfo",
        "/proc/cpuinfo",
        "--ro-bind",
        "/proc/meminfo",
        "/proc/meminfo",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(root / ".venv"),
        "/venv",
        "--bind",
        str(work),
        "/work",
        "--ro-bind",
        str(practice),
        "/practice",
        "--dir",
        "/weights",
        "--dir",
        "/app",
        "--dir",
        "/app/wsp",
        "--clearenv",
        "--setenv",
        "PATH",
        "/venv/bin:/usr/bin",
        "--setenv",
        "HOME",
        "/work",
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "OMP_NUM_THREADS",
        "4",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "4",
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "PYTHONPATH",
        "/app",
        "--chdir",
        "/work",
    ]
    for name in ["__init__.py", "weights.py", "sandbox.py", "python_kernel.py", "resources.py"]:
        command.extend(["--ro-bind", str(root / "src/wsp" / name), f"/app/wsp/{name}"])
    # Resolve each cache symlink without exposing sibling models or credentials.
    for path in sorted(model.iterdir()):
        if path.is_file():
            command.extend(["--ro-bind", str(path.resolve()), f"/weights/{path.name}"])
    return [*command, "/venv/bin/python", "-u", *arguments]
