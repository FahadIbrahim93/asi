"""Contracts for the deterministic GitHub Actions pytest shard planner."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_PLANNER_PATH = _ROOT / ".github" / "scripts" / "plan_pytest_shards.py"


def _load_planner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plan_pytest_shards", _PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_file_counts_rejects_empty_collection() -> None:
    planner = _load_planner()

    with pytest.raises(ValueError, match="selected no test nodes"):
        planner.selected_file_counts(("no tests ran", "tests/test_a.py"))


def test_balancing_is_deterministic_and_exhaustive() -> None:
    planner = _load_planner()
    counts = Counter({"tests/test_a.py": 8, "tests/test_b.py": 5, "tests/test_c.py": 3})

    forward = planner.balanced_file_shards(counts, 2)
    reverse = planner.balanced_file_shards(dict(reversed(tuple(counts.items()))), 2)

    assert forward == reverse
    flattened = [path for files, _total in forward for path in files]
    assert sorted(flattened) == sorted(counts)
    assert len(flattened) == len(set(flattened))
    assert [total for _files, total in forward] == [8, 8]


def test_fewer_files_than_requested_shards_stays_nonempty() -> None:
    planner = _load_planner()

    shards = planner.balanced_file_shards({"tests/test_a.py": 2}, 8)

    assert shards == ((('tests/test_a.py',), 2),)


def test_matrix_expands_the_same_plan_over_each_python_version() -> None:
    planner = _load_planner()
    shards = ((('tests/test_a.py',), 2), (('tests/test_b.py',), 1))

    matrix = planner.shard_matrix(shards, ("3.12", "3.13"))

    assert [(item["python"], item["shard"]) for item in matrix["include"]] == [
        ("3.12", 1),
        ("3.12", 2),
        ("3.13", 1),
        ("3.13", 2),
    ]
    assert json.loads(matrix["include"][0]["files_json"]) == ["tests/test_a.py"]
