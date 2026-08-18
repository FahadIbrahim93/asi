"""Resource contracts for replay and frozen-feature IPMNIST ceilings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

_METHODS = ("replay", "in_context", "randumb", "ranpac", "prol")
_INT32_MAX = 2**31 - 1
_MAX_PERSISTENT_BYTES = 256 * 1024 * 1024
_MAX_FEATURE_DIM = 1_000_000

CEILING_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.ipmnist-ceilings.protocol.v1",
        "paper_revisions": (
            "arXiv:2503.20018v1",
            "arXiv:2402.08823v3",
            "arXiv:2307.02251v3",
            "arXiv:2507.12305v1",
        ),
        "methods": _METHODS,
        "pretraining_allowed_but_charged": True,
        "extractor_queries_charged": True,
        "replay_bytes_charged": True,
        "matched_axes": ("seed", "updates", "observations"),
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _nonnegative(name: str, value: object, *, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


@dataclass(frozen=True)
class CeilingResourceLedger:
    """Complete resource charges, including usually hidden ceiling costs."""

    persistent_bytes: int
    replay_bytes: int
    environment_steps: int
    pretraining_steps: int
    model_queries: int
    extractor_queries: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            maximum = _MAX_PERSISTENT_BYTES if name.endswith("bytes") else _INT32_MAX
            object.__setattr__(
                self, name, _nonnegative(name, getattr(self, name), maximum=maximum)
            )
        if self.persistent_bytes + self.replay_bytes > _MAX_PERSISTENT_BYTES:
            raise ValueError("total persistent bytes exceed 256 MiB")
        if self.environment_steps + self.pretraining_steps > _INT32_MAX:
            raise ValueError("total steps exceed signed int32")
        if self.model_queries + self.extractor_queries > _INT32_MAX:
            raise ValueError("total model queries exceed signed int32")

    @property
    def total_persistent_bytes(self) -> int:
        return self.persistent_bytes + self.replay_bytes

    @property
    def total_steps(self) -> int:
        return self.environment_steps + self.pretraining_steps

    @property
    def total_model_queries(self) -> int:
        return self.model_queries + self.extractor_queries


@dataclass(frozen=True)
class FrozenFeatureCeiling:
    """Prospective frozen-feature/replay arm declaration."""

    method: Literal["replay", "in_context", "randumb", "ranpac", "prol"]
    feature_dim: int
    replay_capacity: int

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _METHODS:
            raise ValueError("method is not a registered ceiling")
        if (
            type(self.feature_dim) is not int
            or self.feature_dim < 1
            or self.feature_dim > _MAX_FEATURE_DIM
        ):
            raise ValueError("feature_dim must be in [1, 1000000]")
        _nonnegative("replay_capacity", self.replay_capacity)

    @property
    def mechanism_off(self) -> bool:
        return self.method == "randumb" and self.replay_capacity == 0

    def persistent_replay_bytes(self, *, example_bytes: int) -> int:
        resolved = _nonnegative(
            "example_bytes", example_bytes, maximum=_MAX_PERSISTENT_BYTES
        )
        if self.replay_capacity and resolved > _MAX_PERSISTENT_BYTES // self.replay_capacity:
            raise ValueError("derived replay bytes exceed 256 MiB")
        return self.replay_capacity * resolved
