"""Strict JSON coverage for runtime and package provenance probes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Never

import pytest

from alberta_framework.benchmarks import _official_foragax_image_helper as image_helper
from alberta_framework.benchmarks import official_foragax

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "payload",
    [
        '{"packages": [], "packages": []}',
        '{"packages": NaN}',
        '{"packages": 1e10000}',
    ],
)
def test_image_runtime_probe_rejects_non_strict_json(
    payload: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=(), returncode=0, stdout=payload, stderr=""
    )
    monkeypatch.setattr(
        "alberta_framework.benchmarks._official_foragax_image_helper.subprocess.run",
        lambda *args, **kwargs: completed,
    )
    with pytest.raises(image_helper.ImageHelperError, match="runtime package probe"):
        image_helper._runtime_probe(Path("unused"))


@pytest.mark.parametrize(
    "payload, message",
    [
        (" " * (image_helper._MAX_RUNTIME_PROBE_BYTES + 1), "byte ceiling"),
        ("[" * 2000 + "]" * 2000, "nesting ceiling"),
    ],
)
def test_image_runtime_probe_rejects_oversized_or_deep_json(
    payload: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess(
        args=(), returncode=0, stdout=payload, stderr=""
    )
    monkeypatch.setattr(
        "alberta_framework.benchmarks._official_foragax_image_helper.subprocess.run",
        lambda *args, **kwargs: completed,
    )
    with pytest.raises(image_helper.ImageHelperError, match=message):
        image_helper._runtime_probe(Path("unused"))


def test_image_runtime_probe_bounds_packages_before_append_and_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["script"] = command[-1]
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"packages":[]}', stderr=""
        )

    monkeypatch.setattr(
        "alberta_framework.benchmarks._official_foragax_image_helper.subprocess.run",
        fake_run,
    )
    assert image_helper._runtime_probe(Path("python")) == {"packages": []}
    script = captured["script"]
    compile(script, "<official-foragax-image-runtime-probe>", "exec")
    assert f"MAX_PACKAGES = {image_helper._MAX_RUNTIME_PACKAGES}" in script
    guard = script.index("len(packages) >= MAX_PACKAGES")
    assert guard < script.index("packages.append(") < script.index("packages.sort(")
    assert script.index("type(name) is not str") < script.index("if not name:")
    assert script.index("type(direct_url) is not str") < script.index("if direct_url and")
    assert "record_text_bytes > MAX_RECORD_TEXT_BYTES" in script
    assert "len(direct_url.encode(\"utf-8\")) > MAX_DIRECT_URL_BYTES" in script


@pytest.mark.parametrize(
    "direct_url",
    [
        '{"url":"https://example.invalid/a","url":"https://example.invalid/b"}',
    ],
)
def test_package_freeze_sanitizer_redacts_non_strict_direct_url(direct_url: str) -> None:
    assert official_foragax._sanitize_package_freeze_line(
        "package==1 ; direct_url=" + direct_url
    ) == "package==1 ; direct_url=<REDACTED>"


def test_strict_json_rejects_inexact_input_without_retaining_payload() -> None:
    inexact: Any = bytearray(b"{}")
    with pytest.raises(
        official_foragax.OfficialForagaxValidationError,
        match="exact JSON text",
    ):
        official_foragax._strict_json_loads(inexact, label="test")
    secret = "private-token"
    with pytest.raises(official_foragax.OfficialForagaxValidationError) as raised:
        official_foragax._strict_json_loads(
            '{"' + secret + '":1,"' + secret + '":2}',
            label="test",
        )
    assert secret not in str(raised.value)


@pytest.mark.parametrize("payload", [b" " * 9, " " * 9])
def test_strict_json_bounds_exact_input_before_scanning(
    payload: bytes | str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(official_foragax, "_MAX_STRICT_JSON_BYTES", 8)
    with pytest.raises(
        official_foragax.OfficialForagaxValidationError,
        match="byte ceiling",
    ):
        official_foragax._strict_json_loads(payload, label="test")


def test_package_freeze_sanitizer_bounds_exact_input_before_parsing() -> None:
    with pytest.raises(official_foragax.OfficialForagaxValidationError, match="byte ceiling"):
        official_foragax._sanitize_package_freeze_line(
            "package==1 ; direct_url=" + " " * official_foragax._MAX_DIRECT_URL_BYTES
        )
    deeply_nested = "[" * 2000 + "]" * 2000
    assert official_foragax._sanitize_package_freeze_line(
        "package==1 ; direct_url=" + deeply_nested
    ) == "package==1 ; direct_url=<REDACTED>"


def test_package_freeze_rejects_oversized_package_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    payload = official_foragax._PROBE_PREFIX.encode() + json.dumps(
        {"packages": [f"package-{index}==1" for index in range(4097)]}
    ).encode()
    completed = subprocess.CompletedProcess(args=(), returncode=0, stdout=payload, stderr=b"")
    def fake_execution(**kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["script"] = str(kwargs["script"])
        return completed

    monkeypatch.setattr(official_foragax, "_run_execution_python", fake_execution)
    with pytest.raises(official_foragax.OfficialForagaxValidationError, match="package freeze"):
        official_foragax._package_freeze(
            repository=Path("."),
            interpreter=Path("python"),
            environment={},
        )
    script = captured["script"]
    compile(script, "<official-foragax-package-freeze>", "exec")
    guard = script.index("len(packages) >= 4096")
    assert guard < script.index('read_text("direct_url.json")')
    assert guard < script.index("packages.append(") < script.index("sorted(set(packages))")
    assert script.index("type(direct_url) is not str") < script.index("if direct_url:")


def test_package_freeze_rejects_cumulative_text_before_sanitizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = "p" * (official_foragax._MAX_DIRECT_URL_BYTES - 1)
    count = official_foragax._MAX_PACKAGE_FREEZE_TEXT_BYTES // len(line.encode()) + 1
    payload = official_foragax._PROBE_PREFIX.encode() + json.dumps(
        {"packages": [line] * count}, separators=(",", ":")
    ).encode()
    completed = subprocess.CompletedProcess(args=(), returncode=0, stdout=payload, stderr=b"")
    monkeypatch.setattr(official_foragax, "_run_execution_python", lambda **_: completed)
    with pytest.raises(official_foragax.OfficialForagaxValidationError, match="byte ceiling"):
        official_foragax._package_freeze(
            repository=Path("."), interpreter=Path("python"), environment={}
        )


def test_generated_runtime_probe_contains_self_contained_strict_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScriptCapturedError(Exception):
        def __init__(self, script: str) -> None:
            self.script = script

    def capture_script(**kwargs: object) -> Never:
        raise ScriptCapturedError(str(kwargs["script"]))

    monkeypatch.setattr(official_foragax, "_run_execution_python", capture_script)
    with pytest.raises(ScriptCapturedError) as raised:
        official_foragax._probe_runtime(
            repository=Path("unused"),
            interpreter=Path("python"),
            environment={},
        )
    script = raised.value.script
    compile(script, "<official-foragax-runtime-probe>", "exec")
    assert "object_pairs_hook=strict_pairs" in script
    assert "parse_float=strict_float" in script
    assert "_strict_json_loads" not in script
    assert "OfficialForagaxValidationError" not in script
    assert '"unparsed": "<REDACTED>"' in script
    assert str(official_foragax._MAX_DIRECT_URL_BYTES) in script
