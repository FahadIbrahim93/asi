"""Strict JSON output contracts for the Step 1/2 smoke CLIs."""

import json
from collections.abc import Callable

import pytest

import alberta_framework.cli as cli
from alberta_framework.cli import _print_json


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cli_json_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        _print_json({"final_window_mse": value, "finite": False})


def test_cli_json_finite_payload_is_strict_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_json({"final_window_mse": 0.125, "finite": True})
    text = capsys.readouterr().out
    assert json.loads(text) == {"final_window_mse": 0.125, "finite": True}
    assert "NaN" not in text
    assert "Infinity" not in text


@pytest.mark.parametrize(
    ("entrypoint", "runner_name"),
    [
        (cli.step1_smoke_main, "run_step1_smoke"),
        (cli.step2_smoke_main, "run_step2_smoke"),
    ],
)
def test_smoke_entrypoint_refuses_nonfinite_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entrypoint: Callable[[list[str]], int],
    runner_name: str,
) -> None:
    class NonfiniteResult:
        finite = False

        def to_dict(self) -> dict[str, object]:
            return {"final_window_mse": float("nan"), "finite": False}

    monkeypatch.setattr(cli, runner_name, lambda *args, **kwargs: NonfiniteResult())
    with pytest.raises(ValueError):
        entrypoint([])
    assert capsys.readouterr().out == ""
