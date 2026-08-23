"""Regression: BoundedPolicyArchive must enforce its declared equal-width
invariant in the constructor, and the single-policy controls must not reject
a descriptor they never read.

Issue #2322: the width guard lived in `add` before the mode short-circuits, so
(1) a directly constructed mixed-width archive violated the declared invariant
and numpy broadcasting fabricated zero distances that silently discarded valid
candidates, and (2) the one_model/fixed_snapshot controls raised on entries
whose latent they never use.
"""

from __future__ import annotations

import pytest

from alberta_framework.core.policy_archive import BoundedPolicyArchive, PolicyEntry

pytestmark = pytest.mark.unit


def _entry(identity: str, latent: tuple[float, ...], score: float = 0.0) -> PolicyEntry:
    return PolicyEntry(
        identity=identity, policy_bytes=b"x", latent=latent, score=score
    )


def test_constructor_rejects_mixed_width_entries() -> None:
    narrow = _entry("a", (0.0,))
    wide = _entry("b", (3.0, 3.0))
    with pytest.raises(ValueError, match="equal width"):
        BoundedPolicyArchive(
            byte_budget=4096, min_latent_distance=1.0, entries=(narrow, wide)
        )


def test_constructor_rejects_mixed_width_in_any_mode() -> None:
    narrow = _entry("a", (0.0,))
    wide = _entry("b", (3.0, 3.0))
    for mode in ("diverse_archive", "one_model", "fixed_snapshot"):
        with pytest.raises(ValueError, match="equal width"):
            BoundedPolicyArchive(
                byte_budget=4096,
                min_latent_distance=1.0,
                mode=mode,
                entries=(narrow, wide),
            )


def test_one_model_control_accepts_wider_entry() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        mode="one_model",
        entries=(_entry("a", (0.0,)),),
    )
    successor = archive.add(_entry("b", (3.0, 3.0), score=9.0))
    assert tuple(e.identity for e in successor.entries) == ("b",)


def test_fixed_snapshot_control_accepts_wider_entry() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        mode="fixed_snapshot",
        entries=(_entry("a", (0.0,)),),
    )
    assert archive.add(_entry("b", (3.0, 3.0), score=9.0)) is archive


def test_diverse_add_still_rejects_mismatched_candidate_width() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        entries=(_entry("a", (0.0,)),),
    )
    with pytest.raises(ValueError, match="equal width"):
        archive.add(_entry("b", (3.0, 3.0)))


def test_diverse_add_retains_distant_candidate() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        entries=(_entry("a", (0.0,)),),
    )
    successor = archive.add(_entry("c", (3.0,), score=1.0))
    assert tuple(e.identity for e in successor.entries) == ("a", "c")


def test_diverse_add_rejects_close_low_score_candidate() -> None:
    archive = BoundedPolicyArchive(
        byte_budget=4096,
        min_latent_distance=1.0,
        entries=(_entry("a", (0.0,), score=5.0),),
    )
    assert archive.add(_entry("c", (0.5,), score=1.0)) is archive
