import pytest

from wsp.resources import require_headroom, stop_owned_process, wait_guarded


def test_resource_reserves_and_request_estimates_are_enforced(monkeypatch, tmp_path):
    gib = 1024**3
    monkeypatch.setattr(
        "wsp.resources.snapshot",
        lambda _: {
            "disk": {"free_bytes": 200 * gib},
            "ram": {"effective_available_bytes": 120 * gib},
        },
    )
    limits = {"minimum_free_disk_gib": 100, "minimum_available_ram_gib": 96}
    require_headroom(tmp_path, limits, disk_bytes=100 * gib, ram_bytes=24 * gib)
    with pytest.raises(RuntimeError, match="Disk guard"):
        require_headroom(tmp_path, limits, disk_bytes=101 * gib)
    with pytest.raises(RuntimeError, match="RAM guard"):
        require_headroom(tmp_path, limits, ram_bytes=25 * gib)
    with pytest.raises(ValueError):
        require_headroom(tmp_path, limits, disk_bytes=-1)


def test_running_worker_is_stopped_when_headroom_disappears(monkeypatch, tmp_path):
    class Child:
        def poll(self):
            return None

    stopped = []
    child = Child()

    def depleted(*args, **kwargs):
        raise RuntimeError("Disk guard: reserve exhausted")

    monkeypatch.setattr("wsp.resources.require_headroom", depleted)
    monkeypatch.setattr("wsp.resources.stop_owned_process", stopped.append)
    with pytest.raises(RuntimeError, match="Disk guard"):
        wait_guarded(child, tmp_path, {})
    assert stopped == [child]


def test_resource_stop_includes_descendants_that_opened_other_process_groups(monkeypatch):
    terminated = []
    waited = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            assert recursive
            return [Process(102), Process(103)]

        def terminate(self):
            terminated.append(self.pid)

        def is_running(self):
            return self.pid not in terminated

    class Child:
        pid = 101

        def poll(self):
            return None

        def wait(self):
            waited.append(self.pid)
            return 0

    monkeypatch.setattr("wsp.resources.psutil.Process", Process)
    stop_owned_process(Child())
    assert terminated == [102, 103, 101]
    assert waited == [101]
