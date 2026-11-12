import httpx
import pytest

from wsp import services
from wsp.io import write_json


@pytest.mark.parametrize("matches", [True, False])
def test_existing_endpoint_must_serve_the_prepared_editor(monkeypatch, matches):
    editor = {"path": "/models/editor", "port": 18801}
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://127.0.0.1:18801/v1/models"),
        json={
            "data": [{"id": "wsp-editor", "root": editor["path"] if matches else "/models/other"}]
        },
    )
    monkeypatch.setattr(services.httpx, "get", lambda *a, **k: response)
    if matches:
        assert services.health(editor)
    else:
        with pytest.raises(RuntimeError, match="different model"):
            services.health(editor)


def test_endpoint_http_errors_are_not_hidden(monkeypatch):
    response = httpx.Response(503, request=httpx.Request("GET", "http://127.0.0.1:18801/v1/models"))
    monkeypatch.setattr(services.httpx, "get", lambda *a, **k: response)
    with pytest.raises(httpx.HTTPStatusError):
        services.health({"path": "/models/editor", "port": 18801})


@pytest.mark.parametrize("changed", ["none", "pid", "command", "model"])
def test_stop_editor_requires_owned_process_identity(tmp_path, monkeypatch, changed):
    record = {"pid": 100000001, "created_at": 42.0, "model": "/models/editor", "port": 18801}
    write_json(tmp_path / "runs/editor_process.json", record)
    stopped = []

    class Process:
        pid = record["pid"]

        def create_time(self):
            return 43.0 if changed == "pid" else record["created_at"]

        def cmdline(self):
            return [
                "python",
                "-m",
                "other_service" if changed == "command" else "vllm.entrypoints.openai.api_server",
                "--model",
                "/models/other" if changed == "model" else record["model"],
                "--served-model-name",
                "wsp-editor",
                "--host",
                "127.0.0.1",
                "--port",
                str(record["port"]),
            ]

    monkeypatch.setattr(services.psutil, "Process", lambda pid: Process())
    monkeypatch.setattr(services, "stop_process_tree", stopped.append)
    if changed == "none":
        services.stop_editor(tmp_path)
        assert stopped == [record["pid"]]
    else:
        with pytest.raises(RuntimeError):
            services.stop_editor(tmp_path)
        assert not stopped
