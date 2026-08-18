"""Byte-bounded immutable policy archive and single-policy controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import numpy as np

POLICY_ARCHIVE_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.policy-archive.protocol.v1",
        "paper_revision": "arXiv:2604.15414v1",
        "paper_revision_date": "2026-04-16",
        "adaptation_difference": "generic latent descriptors; no MiniGrid/task archive claim",
        "controls": ("one_model", "fixed_snapshot"),
        "matched_axes": ("seed", "updates", "observations", "policy_queries"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

ArchiveMode = Literal["diverse_archive", "one_model", "fixed_snapshot"]


@dataclass(frozen=True)
class PolicyEntry:
    identity: str
    policy_bytes: bytes
    latent: tuple[float, ...]
    score: float

    def __post_init__(self) -> None:
        if type(self.identity) is not str or not self.identity:
            raise ValueError("identity must be a non-empty string")
        if type(self.policy_bytes) is not bytes or not self.policy_bytes:
            raise ValueError("policy_bytes must be non-empty exact bytes")
        if type(self.latent) is not tuple or not self.latent:
            raise ValueError("latent must be a non-empty tuple")
        if any(type(value) is not float or not math.isfinite(value) for value in self.latent):
            raise ValueError("latent values must be finite floats")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise ValueError("score must be a finite float")

    @property
    def persistent_bytes(self) -> int:
        """Canonical retained payload: identity UTF-8, policy, latent, and score."""
        return (
            len(self.identity.encode("utf-8"))
            + len(self.policy_bytes)
            + 8 * len(self.latent)
            + 8
        )


@dataclass(frozen=True)
class BoundedPolicyArchive:
    """Pure archive reducer with an exact serialized-policy byte ceiling."""

    byte_budget: int
    min_latent_distance: float
    mode: ArchiveMode = "diverse_archive"
    entries: tuple[PolicyEntry, ...] = ()

    def __post_init__(self) -> None:
        if type(self.byte_budget) is not int or self.byte_budget < 1:
            raise ValueError("byte_budget must be a positive integer")
        if (
            type(self.min_latent_distance) is not float
            or not math.isfinite(self.min_latent_distance)
            or self.min_latent_distance < 0.0
        ):
            raise ValueError("min_latent_distance must be a finite non-negative float")
        if type(self.mode) is not str or self.mode not in {
            "diverse_archive",
            "one_model",
            "fixed_snapshot",
        }:
            raise ValueError("unknown archive mode")
        if type(self.entries) is not tuple or any(
            type(entry) is not PolicyEntry for entry in self.entries
        ):
            raise ValueError("entries must be an exact tuple of PolicyEntry values")
        if len({entry.identity for entry in self.entries}) != len(self.entries):
            raise ValueError("entry identities must be unique")
        if self.persistent_bytes > self.byte_budget:
            raise ValueError("entries exceed byte budget")

    @property
    def persistent_bytes(self) -> int:
        return sum(entry.persistent_bytes for entry in self.entries)

    def add(self, entry: PolicyEntry) -> BoundedPolicyArchive:
        """Return the deterministic successor archive."""
        if type(entry) is not PolicyEntry:
            raise ValueError("entry must be an exact PolicyEntry")
        if any(old.identity == entry.identity for old in self.entries):
            raise ValueError("entry identity already exists")
        candidates: tuple[PolicyEntry, ...]
        if self.entries and len(entry.latent) != len(self.entries[0].latent):
            raise ValueError("all latent descriptors must have equal width")
        if self.mode == "fixed_snapshot" and self.entries:
            return self
        if self.mode == "one_model":
            candidates = (entry,)
        elif not self.entries:
            candidates = (entry,)
        else:
            distances = tuple(
                float(np.linalg.norm(np.asarray(entry.latent) - np.asarray(old.latent)))
                for old in self.entries
            )
            nearest = int(np.argmin(np.asarray(distances)))
            if distances[nearest] < self.min_latent_distance:
                if entry.score <= self.entries[nearest].score:
                    return self
                candidates = self.entries[:nearest] + (entry,) + self.entries[nearest + 1 :]
            else:
                candidates = self.entries + (entry,)
        if sum(item.persistent_bytes for item in candidates) > self.byte_budget:
            raise ValueError("candidate would exceed byte budget")
        return replace(self, entries=candidates)
