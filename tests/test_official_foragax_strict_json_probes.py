"""Strict JSON admission for official Foragax runtime metadata."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from alberta_framework.benchmarks import _official_foragax_image_helper as image_helper
from alberta_framework.benchmarks import official_foragax


def test_image_runtime_probe_rejects_duplicate_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=("python",),
        returncode=0,
        stdout='{"packages":[],"packages":{}}',
        stderr="",
    )
    monkeypatch.setattr(image_helper.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(image_helper.ImageHelperError, match="repeats JSON key"):
        image_helper._runtime_probe(Path("/trusted/python"))


def test_package_freeze_direct_url_rejects_duplicate_keys() -> None:
    with pytest.raises(
        official_foragax.OfficialForagaxValidationError,
        match="duplicate object key",
    ):
        official_foragax._sanitize_package_freeze_line(
            'pkg==1.0 ; direct_url={"url":"https://first","url":"https://second"}'
        )


def test_generated_distribution_probe_uses_strict_direct_url_decoder() -> None:
    namespace: dict[str, object] = {"json": json, "math": math}
    exec(official_foragax._STRICT_DIRECT_URL_HELPER_SOURCE, namespace)
    loads = namespace["_strict_direct_url_loads"]
    assert callable(loads)
    assert loads('{"url":"https://example.test"}') == {"url": "https://example.test"}
    with pytest.raises(ValueError, match="duplicate key"):
        loads('{"url":"https://first","url":"https://second"}')
    with pytest.raises(ValueError, match="non-finite"):
        loads('{"weight":1e999}')


def test_runtime_probe_embeds_and_calls_the_strict_direct_url_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_execution(**kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["script"] = str(kwargs["script"])
        return subprocess.CompletedProcess(("python",), 0, b"", b"")

    monkeypatch.setattr(official_foragax, "_run_execution_python", fake_execution)
    with pytest.raises(
        official_foragax.OfficialForagaxValidationError,
        match="did not return probe metadata",
    ):
        official_foragax._probe_runtime(
            repository=Path("/trusted/repository"),
            interpreter=Path("/trusted/python"),
            environment={},
        )

    assert official_foragax._STRICT_DIRECT_URL_HELPER_SOURCE in captured["script"]
    assert "direct_url = _strict_direct_url_loads(direct_url_text)" in captured["script"]
