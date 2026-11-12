import csv
import io
import shutil
import subprocess
import time
from pathlib import Path

import psutil


def require_gpu_headroom(gpus: list[int], fraction: float, reserve_gib: float = 8) -> None:
    if (
        not gpus
        or len(gpus) != len(set(gpus))
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
    ):
        raise ValueError("Use distinct nonnegative physical GPU indices.")
    if not 0 < fraction < 1:
        raise ValueError("GPU memory utilization must be between zero and one.")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory = {
        int(index): (float(total), float(free))
        for index, total, free in csv.reader(io.StringIO(output))
    }
    for gpu in gpus:
        if gpu not in memory:
            raise ValueError(f"Physical GPU {gpu} does not exist.")
        total, free = memory[gpu]
        required = total * fraction + reserve_gib * 1024
        if free < required:
            raise RuntimeError(
                f"GPU {gpu} needs {required:.0f} MiB free; only {free:.0f} available."
            )


def memory_status() -> dict:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            fields[name] = int(value.split()[0]) * 1024
    available = fields["MemAvailable"]
    constraints = []
    root = Path("/sys/fs/cgroup")
    if (root / "cgroup.controllers").is_file():
        for directory in [root]:
            limit_file, current_file = directory / "memory.max", directory / "memory.current"
            if limit_file.is_file() and current_file.is_file():
                raw = limit_file.read_text().strip()
                if raw != "max":
                    limit, used = int(raw), int(current_file.read_text())
                    available = min(available, max(0, limit - used))
                    constraints.append({"limit_bytes": limit, "used_bytes": used})
    else:
        limit_file = root / "memory/memory.limit_in_bytes"
        used_file = root / "memory/memory.usage_in_bytes"
        if limit_file.is_file() and used_file.is_file():
            limit, used = int(limit_file.read_text()), int(used_file.read_text())
            available = min(available, max(0, limit - used))
            constraints.append({"limit_bytes": limit, "used_bytes": used})
    return {
        "host_total_bytes": fields["MemTotal"],
        "host_available_bytes": fields["MemAvailable"],
        "cgroup_limits": constraints,
        "effective_available_bytes": available,
    }


def snapshot(root: Path) -> dict:
    disk = shutil.disk_usage(root)
    return {
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "ram": memory_status(),
    }


def require_headroom(root: Path, limits: dict, disk_bytes: int = 0, ram_bytes: int = 0) -> dict:
    if disk_bytes < 0 or ram_bytes < 0:
        raise ValueError("Resource estimates cannot be negative.")
    measured = snapshot(root)
    disk_required = disk_bytes + int(limits["minimum_free_disk_gib"] * 1024**3)
    ram_required = ram_bytes + int(limits["minimum_available_ram_gib"] * 1024**3)
    if measured["disk"]["free_bytes"] < disk_required:
        raise RuntimeError(
            f"Disk guard: need {disk_required / 1024**3:.1f} GiB free including reserve; "
            f"have {measured['disk']['free_bytes'] / 1024**3:.1f} GiB. Work stopped."
        )
    if measured["ram"]["effective_available_bytes"] < ram_required:
        raise RuntimeError(
            f"RAM guard: need {ram_required / 1024**3:.1f} GiB including reserve; "
            f"have {measured['ram']['effective_available_bytes'] / 1024**3:.1f} GiB. "
            "Work stopped."
        )
    return measured


def stop_owned_process(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    stop_process_tree(child.pid)
    child.wait()


def stop_process_tree(pid: int) -> None:
    """Stop a caller-verified PID and its descendants, never processes selected by name."""
    if pid <= 1:
        raise ValueError("Refusing to stop a system process.")
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    alive = _wait_for_process_exit(processes, timeout=20)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    remaining = _wait_for_process_exit(alive, timeout=10)
    if remaining:
        raise TimeoutError(
            f"Owned processes did not stop: {[process.pid for process in remaining]}"
        )


def _wait_for_process_exit(processes: list[psutil.Process], timeout: float) -> list[psutil.Process]:
    deadline = time.monotonic() + timeout
    pending = list(processes)
    while pending:
        alive = []
        for process in pending:
            try:
                if process.is_running() and process.status() not in {
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                }:
                    alive.append(process)
            except psutil.NoSuchProcess:
                continue
        if not alive or time.monotonic() >= deadline:
            return alive
        pending = alive
        time.sleep(0.1)
    return []


def wait_guarded(
    child: subprocess.Popen,
    root: Path,
    limits: dict,
    timeout: int | None = None,
) -> int:
    started = time.monotonic()
    try:
        while child.poll() is None:
            require_headroom(root, limits)
            if timeout is not None and time.monotonic() - started > timeout:
                raise subprocess.TimeoutExpired(child.args, timeout)
            try:
                return child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
        return child.wait()
    finally:
        stop_owned_process(child)
