import ast

import pytest

from wsp.python_kernel import execute_cell, execute_persistent
from wsp.sandbox import bootstrap_code


def test_variables_functions_and_last_expression_survive_cells(capsys):
    state = {}
    execute_cell("x = [1, 2]\ndef total():\n    return sum(x)\nx", state)
    execute_cell("x.append(3)\ntotal()", state)
    assert capsys.readouterr().out == "[1, 2]\n6\n"
    assert state["x"] == [1, 2, 3]


def test_code_errors_do_not_erase_prior_variables():
    state = {}
    execute_cell("x = 7", state)
    with pytest.raises(ZeroDivisionError):
        execute_cell("1 / 0", state)
    assert state["x"] == 7


def test_training_guard_still_applies_in_persistent_cells():
    with pytest.raises(ValueError, match="backward"):
        execute_cell("loss.backward()", {})


def test_sandbox_bootstrap_passes_the_output_policy():
    tree = ast.parse(bootstrap_code())
    calls = [
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "WeightWorkspace"
    ]
    keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    assert keywords == {}


def test_persistent_executor_propagates_policy_without_resetting_state(tmp_path, monkeypatch):
    received = []

    def kernel(*args):
        received.append(args)
        return {"kernel_id": "same"}, False

    monkeypatch.setattr("wsp.python_kernel.ensure_kernel", kernel)
    monkeypatch.setattr(
        "wsp.python_kernel.exchange", lambda *args: {"kernel_id": "same", "exit_code": 0}
    )
    result = execute_persistent(
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        "pass",
        1,
    )
    assert received[0] == (tmp_path, tmp_path, tmp_path, tmp_path)
    assert result["kernel_restarted"] is False
