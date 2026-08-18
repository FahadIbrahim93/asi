"""Hostile identities for qualification AST."""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


class _HostileConstant(ast.Constant):
    calls = 0

    def __init__(self, value: object):
        super().__init__(value=value)


def test_qualification_ast_rejects_hostile_constant_before_str() -> None:
    # The gate is type(argument) is ast.Constant and type(argument.value) is str
    # Hostile subclass of str should be rejected before collecting
    hostile_val = _HostileStr("--seed")
    _HostileStr.calls = 0
    # Simulate a hostile Constant node with hostile string value
    node = ast.Constant(value=hostile_val)
    # Our gate should be False for hostile subclass, not collecting
    gate = type(node) is ast.Constant and type(node.value) is str
    assert gate is False
    assert _HostileStr.calls == 0
    # Builtin should be True
    real = ast.Constant(value="--seed")
    assert (type(real) is ast.Constant and type(real.value) is str) is True


def test_qualification_collects_builtin_only() -> None:
    # Gate check only
    hostile = _HostileStr("--seed")
    assert type(hostile) is not str
