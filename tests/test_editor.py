import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from test_action_transport import Tokenizer

from wsp import autonomy
from wsp.config import ROOT, initial_messages
from wsp.editor import Experiment, capture_inputs, practice_selection
from wsp.io import read_json, read_json_gzip, read_jsonl, write_json, write_jsonl
from wsp.weights import WeightWorkspace

MATRIX = "model.language_model.layers.0.mlp.down_proj.weight"


@pytest.fixture
def root(tmp_path):
    for path in (ROOT / "configs").glob("*"):
        destination = tmp_path / "configs" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
    for name in ["pyproject.toml", "uv.lock", "scripts/run.sh"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    models = read_json(tmp_path / "configs/models.json")
    for role, value in models.items():
        path = tmp_path / role
        path.mkdir()
        value["path"] = str(path)
    save_file(
        {
            MATRIX: torch.arange(16).reshape(4, 4).bfloat16(),
            "model.language_model.embed_tokens.weight": torch.ones(3, 4).bfloat16(),
            "model.language_model.norm.weight": torch.ones(4).bfloat16(),
        },
        tmp_path / "target/model.safetensors",
    )
    write_json(tmp_path / "runs/models.json", models)
    for name in read_json(tmp_path / "configs/data.json")["practice"]["files"]:
        write_jsonl(
            tmp_path / "runs/data/practice" / name, [{"id": name, "problem": "1+1?", "answer": "2"}]
        )
    return tmp_path


def record_turn(experiment, turn):
    record = {
        "turn": turn,
        "content": '{"action":"python","body":"pass"}',
        "reasoning_content": "Own reasoning",
        "tool_result": {"exit_code": 0},
    }
    write_json(experiment.folder / "turns" / f"{turn:04d}.json", record)
    experiment.messages = autonomy.recorded_messages(experiment.folder, turn)
    experiment.turn = turn


def test_research_starts_fresh_and_rejects_unrecorded_guidance(root):
    experiment = Experiment(root, "first")
    assert list(experiment.work.iterdir()) == []
    assert experiment.messages == initial_messages(root)
    with pytest.raises(RuntimeError, match="Unrecorded instructions"):
        autonomy.audit(
            root,
            experiment.folder,
            experiment.messages + [{"role": "user", "content": "Try another edit"}],
            0,
        )
    (root / "configs/user_prompt.txt").write_text("Changed scientific advice")
    with pytest.raises(RuntimeError, match="sealed prompt"):
        Experiment(root, "first")


def test_model_handoff_preserves_its_own_notes_and_initial_prompt(root):
    experiment = Experiment(root, "handoff")
    for turn in range(1, 41):
        record_turn(experiment, turn)
    before = copy.deepcopy(experiment.messages)
    assert autonomy.handoff(root, experiment.folder, before, 40) is None
    notes = "My own observed results and choice of next experiment."
    (experiment.work / "research_notes.txt").write_text(notes)
    after = autonomy.handoff(root, experiment.folder, before, 40)
    assert after[:2] == initial_messages(root) and notes in after[-1]["content"]
    assert read_json_gzip(experiment.folder / "handoffs/0040_history.json.gz") == before
    autonomy.audit(root, experiment.folder, after, 40)
    value = read_json(experiment.folder / "handoffs/0040.json")
    value["content"] += "\nController suggestion"
    write_json(experiment.folder / "handoffs/0040.json", value)
    with pytest.raises(RuntimeError, match="handoff history"):
        autonomy.audit(root, experiment.folder, after, 40)


def test_pending_action_and_changed_original_are_not_silently_restarted(root):
    experiment = Experiment(root, "pending")
    write_json(experiment.folder / "turns/0001.json", {"turn": 1})
    with pytest.raises(RuntimeError, match="interrupted proposal"):
        Experiment(root, "pending")
    (experiment.folder / "turns/0001.json").unlink()
    (experiment.base / "model.safetensors").write_bytes(b"different original")
    with pytest.raises(RuntimeError, match="original checkpoint"):
        Experiment(root, "pending")


def test_context_files_cannot_escape_the_workspace(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text('["not available"]')
    (work / "link.json").symlink_to(outside)
    for value in ["/etc/passwd", "/work/../secret.json", "/work/link.json"]:
        with pytest.raises(ValueError):
            capture_inputs({"contexts_file": value}, work)
    write_json(work / "texts.json", ["Own continuation"])
    assert capture_inputs({"contexts_file": "/work/texts.json"}, work) == {
        "texts": ["Own continuation"]
    }


@pytest.fixture
def model_api(monkeypatch):
    requests = []
    actions = []

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, path, **kwargs):
            requests.append(kwargs["json"])
            action = actions.pop(0)
            content = json.dumps(action)
            tokenizer = Tokenizer()
            tokens = (
                list(map(ord, "Own reasoning"))
                + [tokenizer.convert_tokens_to_ids("</think>")]
                + list(map(ord, content))
                + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )
            return Response(
                {
                    "choices": [
                        {
                            "message": {"content": content},
                            "token_ids": tokens if action is not None else [],
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": len(tokens)},
                }
            )

    monkeypatch.setattr("wsp.editor.httpx.Client", Client)
    monkeypatch.setattr("wsp.editor.ensure_editor", lambda *a: None)
    monkeypatch.setattr("wsp.editor.require_headroom", lambda *a, **k: {})
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    return actions, requests


def test_real_loop_preserves_reasoning_and_only_model_final_can_be_evaluated(
    root, monkeypatch, model_api
):
    experiment = Experiment(root, "loop")
    actions, requests = model_api
    actions.extend(
        [
            {"action": "python", "body": "make_own_edit"},
            {"action": "final", "body": "My measured candidate."},
        ]
    )

    def execute(root, work, base, practice, code, turn, **kwargs):
        workspace = WeightWorkspace(base, work)
        weights = workspace.load_original(MATRIX)
        weights[0, 0] = 0.125
        workspace.save_edit(MATRIX, weights)
        return {"exit_code": 0, "output": "Saved own edit"}

    monkeypatch.setattr("wsp.editor.execute_persistent", execute)
    with pytest.raises(RuntimeError, match="blocked until"):
        autonomy.require_final(root, experiment.folder)
    experiment.run(max_steps=1)
    assert experiment.turn == 1
    resumed = Experiment(root, "loop")
    assert resumed.messages == experiment.messages
    resumed.run(max_steps=1)
    final = autonomy.require_final(root, experiment.folder)
    assert final["edit"]["tensors"][0]["name"] == MATRIX
    assert final["selection"] == "model_final"
    assert all(request["max_tokens"] == 393216 for request in requests)
    assert requests[1]["messages"][2]["reasoning_content"] == "Own reasoning"
    with pytest.raises(ValueError, match="finalized"):
        Experiment(root, "loop")


@pytest.mark.parametrize("action", [None, {"action": "invalid", "body": "pass"}])
def test_two_malformed_replies_stop_without_selecting_weights(root, model_api, action):
    actions, requests = model_api
    actions.extend([action, action])
    experiment = Experiment(root, "malformed")
    with pytest.raises(RuntimeError, match="Two incomplete"):
        experiment.run()
    assert len(requests) == 2 and experiment.turn == 2
    assert not (experiment.folder / "final.json").exists()
    autonomy.audit(root, experiment.folder, experiment.messages, experiment.turn)


def test_valid_tool_errors_are_returned_without_becoming_protocol_failures(root, model_api):
    actions, requests = model_api
    actions.extend(
        [
            {"action": "forward", "body": {"type": "generate", "model": ["original"]}},
            {"action": "forward", "body": {"type": "teacher_forced", "contexts_file": None}},
            {"action": "final", "body": "I did not find an edit."},
        ]
    )
    experiment = Experiment(root, "invalid-inputs")
    experiment.run()
    assert len(requests) == 3
    assert read_json(experiment.folder / "final.json")["status"] == "no_edit"
    with pytest.raises(RuntimeError, match="without an edit"):
        autonomy.require_final(root, experiment.folder)
    for turn in [1, 2]:
        record = read_json(experiment.folder / "turns" / f"{turn:04d}.json")
        assert record["tool_result"]["error"] == "ValueError"


@pytest.mark.parametrize("value", [None, [], {}, 42])
def test_invalid_tool_field_types_have_actionable_errors(tmp_path, value):
    with pytest.raises(ValueError, match="Generation model"):
        practice_selection({"model": value}, {})
    with pytest.raises(ValueError, match="contexts_file"):
        capture_inputs({"contexts_file": value}, tmp_path)


def test_generation_tool_keeps_fixed_sampling_and_saves_exact_candidate(root, monkeypatch):
    experiment = Experiment(root, "generate")
    workspace = WeightWorkspace(experiment.base, experiment.work)
    tensor = workspace.load_original(MATRIX)
    tensor[0, 0] = 0.125
    workspace.save_edit(MATRIX, tensor)
    calls = []

    def worker(module, config, label):
        from wsp.grading import summarize

        calls.append(config)
        assert module == "wsp.evaluate"
        assert config["sampling"]["max_tokens"] == 32768
        problem = read_jsonl(Path(config["dataset"]))[0]
        rows = [
            {
                "id": problem["id"],
                "sample": 0,
                "correct": True,
                "tokens": 10,
                "finish_reason": "stop",
                "thinking_closed": True,
                "grading_status": "graded",
            }
        ]
        output = Path(config["output"])
        write_jsonl(output / "per_sample.jsonl", rows)
        write_jsonl(output / "generations.jsonl", rows)
        write_json(output / "summary.json", {"metrics": summarize(rows)})

    monkeypatch.setattr(experiment, "worker", worker)
    result = experiment.forward(
        {
            "type": "generate",
            "model": "both",
            "problems": [next(iter(experiment.practice))],
            "max_tokens": 32768,
        }
    )
    assert len(calls) == 2 and result["results"]["paired"]["accuracy_difference"] == 0
    assert (experiment.folder / "candidates/fwd_1_attempt_1/delta.npz").is_file()
    assert not Path(calls[1]["model"]).exists()
    assert len(result["files"]) == 2
