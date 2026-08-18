import pytest

from alberta_framework.core.policy_archive import (
    POLICY_ARCHIVE_PROTOCOL,
    BoundedPolicyArchive,
    PolicyEntry,
)


def _entry(name: str, latent: tuple[float, ...], score: float, size: int = 4) -> PolicyEntry:
    return PolicyEntry(identity=name, policy_bytes=bytes(size), latent=latent, score=score)


def test_diverse_archive_respects_exact_byte_budget() -> None:
    archive = BoundedPolicyArchive(byte_budget=8, min_latent_distance=0.5)
    archive = archive.add(_entry("a", (0.0, 0.0), 1.0))
    archive = archive.add(_entry("b", (1.0, 0.0), 2.0))
    assert archive.persistent_bytes == 8
    with pytest.raises(ValueError, match="byte budget"):
        archive.add(_entry("c", (0.0, 1.0), 3.0))


def test_nearby_policy_only_replaces_on_higher_score() -> None:
    archive = BoundedPolicyArchive(byte_budget=8, min_latent_distance=0.5).add(
        _entry("a", (0.0,), 1.0)
    )
    assert archive.add(_entry("low", (0.1,), 0.5)) == archive
    improved = archive.add(_entry("high", (0.1,), 2.0))
    assert [entry.identity for entry in improved.entries] == ["high"]


def test_one_model_and_fixed_snapshot_controls() -> None:
    one = BoundedPolicyArchive(byte_budget=4, min_latent_distance=0.0, mode="one_model")
    one = one.add(_entry("a", (0.0,), 1.0)).add(_entry("b", (1.0,), 0.0))
    assert [entry.identity for entry in one.entries] == ["b"]
    fixed = BoundedPolicyArchive(byte_budget=4, min_latent_distance=0.0, mode="fixed_snapshot")
    fixed = fixed.add(_entry("a", (0.0,), 1.0)).add(_entry("b", (1.0,), 2.0))
    assert [entry.identity for entry in fixed.entries] == ["a"]


def test_protocol_is_nonpromoting() -> None:
    assert POLICY_ARCHIVE_PROTOCOL["paper_revision"] == "arXiv:2604.15414v1"
    assert POLICY_ARCHIVE_PROTOCOL["controls"] == ("one_model", "fixed_snapshot")
    assert POLICY_ARCHIVE_PROTOCOL["scientific_promotion_allowed"] is False
