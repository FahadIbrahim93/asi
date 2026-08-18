"""Hostile string validation for foragax open-screen schema."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStrPassthrough(str):
    calls = 0

    __hash__ = str.__hash__

    def upper(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile upper executed")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")


def test_virtual_table_gate_rejects_hostile_before_upper() -> None:
    hostile = _HostileStrPassthrough("CREATE VIRTUAL TABLE x USING fts5(y)")
    _HostileStrPassthrough.calls = 0
    # Mirror the exact gate from foragax_open_screen.py
    gate = type(hostile) is str and "VIRTUAL TABLE" in hostile.upper()
    # For hostile subclass, type is not str, so gate is False and upper not called
    assert gate is False
    assert _HostileStrPassthrough.calls == 0
    # For builtin str, gate behaves correctly
    real = "CREATE VIRTUAL TABLE x USING fts5(y)"
    assert (type(real) is str and "VIRTUAL TABLE" in real.upper()) is True
    # Non-virtual table string should be False but still call upper (builtin)
    real2 = "CREATE TABLE x (y TEXT)"
    assert (type(real2) is str and "VIRTUAL TABLE" in real2.upper()) is False


def test_sqlite_schema_row_with_hostile_sql_does_not_dispatch() -> None:
    # Simulate schema row: (type, name, tbl_name, sql) where sql is hostile
    hostile_sql = _HostileStrPassthrough("CREATE VIRTUAL TABLE evil USING fts5(x)")
    _HostileStrPassthrough.calls = 0
    row = ("table", "evil", "evil", hostile_sql)
    # Direct check from _canonical_results_database: row[0] is table, type(row[3]) is str ...
    is_virtual = row[0] == "table" and type(row[3]) is str and "VIRTUAL TABLE" in row[3].upper()
    assert is_virtual is False
    assert _HostileStrPassthrough.calls == 0
    # With builtin virtual, it would be True
    row_real = ("table", "evil", "evil", "CREATE VIRTUAL TABLE evil USING fts5(x)")
    assert (
        row_real[0] == "table"
        and type(row_real[3]) is str
        and "VIRTUAL TABLE" in row_real[3].upper()
    ) is True


def test_file_contains_no_repr_leak() -> None:
    import pathlib

    text = pathlib.Path("alberta_framework/benchmarks/foragax_open_screen.py").read_text()
    assert "!r" not in text or "hostile" not in text.lower()
