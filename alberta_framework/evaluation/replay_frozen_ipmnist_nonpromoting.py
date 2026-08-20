"""Fail-closed receipts for replay and frozen-feature IPMNIST controls."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

from alberta_framework.benchmarks.replay_frozen_ipmnist import (
    FROZEN_FEATURE_DIM,
    PROL_COMMIT,
    PROL_PAPER_REVISION,
    RANDUMB_COMMIT,
    RANDUMB_PAPER_REVISION,
    RANPAC_COMMIT,
    RANPAC_PAPER_REVISION,
    REPLAY_CAPACITY,
    REPLAY_OFFICIAL_CODE,
    REPLAY_PAPER_REVISION,
    frozen_hyperparameters,
    replay_hyperparameters,
)

RESULT_SCHEMA: Final = "asi.replay-frozen-ipmnist.development-result.v1"
COMPARISON_ID: Final = "asi.replay-frozen-ipmnist.current-runner.v1"
DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

PROTOCOL_GAPS: Final = (
    "all arms use ASI's online IPMNIST stream and pre-update metric",
    "the replay paper's released arXiv v1 discloses no official implementation",
    "bounded label attention is an in-context proxy, not the paper's trained Transformer",
    "the replay buffer contains 32 raw prior examples rather than paper task-specific memories",
    "RanDumb uses raw IPMNIST pixels rather than the paper's benchmark transforms",
    "RanPAC uses a random raw-pixel extractor because no pretrained ViT is imported",
    "the RanPAC proxy uses recursive ridge rather than task-end Gram recomputation",
    "PROL is an architecture-only prompt/affine proxy without its pretrained model",
    "PROL class-incremental hard-soft semantics are adapted to task-free label streaming",
    "the runner retains an inert protocol MLP as charged compatibility ballast for frozen arms",
    "pretraining examples, updates, and bytes are exactly zero for every local arm",
    "losses across cross-entropy and recursive-readout arms are not directly comparable",
    "development outcomes are retained and can never promote or support external-paper claims",
)

PROTOCOL: Final[Mapping[str, object]] = MappingProxyType(
    {
        "papers": (
            REPLAY_PAPER_REVISION,
            RANDUMB_PAPER_REVISION,
            RANPAC_PAPER_REVISION,
            PROL_PAPER_REVISION,
        ),
        "official_code": {
            "replay": REPLAY_OFFICIAL_CODE,
            "randumb": ("https://github.com/drimpossible/RanDumb", RANDUMB_COMMIT),
            "ranpac": ("https://github.com/McDonnell-Research-Lab/RanPAC", RANPAC_COMMIT),
            "prol": ("https://github.com/anwarmaxsum/PROL", PROL_COMMIT),
        },
        "protocol_gaps": PROTOCOL_GAPS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcome_retention_required": True,
    }
)

_ARMS: Final = MappingProxyType(
    {
        "replay_context_mechanism_off": (
            "replay",
            replay_hyperparameters(replay_update=0.0, context=0.0),
        ),
        "replay_gradient_only": ("replay", replay_hyperparameters(replay_update=1.0, context=0.0)),
        "replay_context_only": ("replay", replay_hyperparameters(replay_update=0.0, context=1.0)),
        "replay_context_full": ("replay", replay_hyperparameters(replay_update=1.0, context=1.0)),
        "randumb_random_features": ("randumb", frozen_hyperparameters(method=0.0, mechanism=1.0)),
        "ranpac_random_projection": ("ranpac", frozen_hyperparameters(method=1.0, mechanism=1.0)),
        "prol_prompt_mechanism_off": ("prol", frozen_hyperparameters(method=2.0, mechanism=0.0)),
        "prol_prompt_proxy": ("prol", frozen_hyperparameters(method=2.0, mechanism=1.0)),
    }
)

_INT32_MAX = (1 << 31) - 1
_MAX_BYTES = 256 * 1024 * 1024
_PARAMETER_LEAVES = 6


def registered_arms() -> tuple[str, ...]:
    """Return the exact replay/frozen campaign roster in execution order."""

    return tuple(_ARMS)


def _object(value: object, keys: frozenset[str], context: str) -> Mapping[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{context} must contain exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _int(value: object, context: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{context} must be an exact bounded integer")
    return value


def _float(value: object, context: str, maximum: float = 3.4028234663852886e38) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= maximum:
        raise ValueError(f"{context} must be an exact bounded nonnegative float")
    return value


def _strings(value: object, context: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or len(value) > 32
        or any(
            type(item) is not str or not item or "\x00" in item or len(item.encode()) > 512
            for item in value
        )
    ):
        raise ValueError(f"{context} must be a bounded exact string list")
    result = tuple(cast(list[str], value))
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


def _parameter_count(input_dim: int, hidden1: int, hidden2: int, classes: int) -> int:
    products = (input_dim * hidden1, hidden1 * hidden2, hidden2 * classes)
    if any(value > _INT32_MAX for value in products):
        raise ValueError("parameter shape exceeds signed int32")
    result = sum(products) + hidden1 + hidden2 + classes
    if result > _INT32_MAX:
        raise ValueError("parameter count exceeds signed int32")
    return result


def _expected_resources(
    family: str,
    observations: int,
    input_dim: int,
    hidden1: int,
    hidden2: int,
    classes: int,
) -> dict[str, int]:
    parameters = _parameter_count(input_dim, hidden1, hidden2, classes)
    if family == "replay":
        persistent_scalars = (
            3 * parameters
            + 5 * _PARAMETER_LEAVES
            + REPLAY_CAPACITY * input_dim
            + REPLAY_CAPACITY
            + 3
        )
        extractor_bytes = 0
        replay_bytes = 4 * (REPLAY_CAPACITY * input_dim + REPLAY_CAPACITY + 2)
        extractor_queries = 0
        replay_queries = REPLAY_CAPACITY * observations
        attention = 2 * REPLAY_CAPACITY * observations
    else:
        m = FROZEN_FEATURE_DIM
        persistent_scalars = parameters + m * input_dim + m + m * m + m * classes + 3 * m + 1
        extractor_bytes = 4 * (m * input_dim + m)
        replay_bytes = 0
        extractor_queries = 2 * observations
        replay_queries = 0
        attention = 0
    values = {
        "persistent_bytes": 4 * persistent_scalars,
        "replay_bytes": replay_bytes,
        "extractor_parameter_bytes": extractor_bytes,
        "pretraining_bytes": 0,
        "environment_steps": 0,
        "data_steps": observations,
        "optimizer_updates": observations,
        "replay_model_queries": replay_queries,
        "extractor_queries": extractor_queries,
        "attention_dot_products": attention,
        "pretraining_examples": 0,
        "pretraining_optimizer_updates": 0,
        "pretraining_model_queries": 0,
        # Contextual prediction, the independently differentiated current loss,
        # and post-update prediction are three distinct MLP forwards.
        "model_queries": 3 * observations + replay_queries,
    }
    if any(value > _INT32_MAX for value in values.values()):
        raise ValueError("derived resource counter exceeds signed int32")
    if values["persistent_bytes"] > _MAX_BYTES:
        raise ValueError("persistent state exceeds the 256 MiB lane bound")
    return values


_OUTER_KEYS = frozenset(
    {
        "schema",
        "comparison_id",
        "paper_revisions",
        "official_commits",
        "protocol_gaps",
        "arm",
        "family",
        "seed",
        "n_tasks",
        "task_length",
        "input_dim",
        "hidden1",
        "hidden2",
        "n_classes",
        "observations",
        "allowed_boundary_information",
        "allowed_task_information",
        "hyperparameters",
        "metrics",
        "resources",
        "outcome",
        "negative_outcome_retained",
        "development_only",
        "scientific_promotion_allowed",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "persistent_bytes",
        "replay_bytes",
        "extractor_parameter_bytes",
        "pretraining_bytes",
        "environment_steps",
        "data_steps",
        "optimizer_updates",
        "replay_model_queries",
        "extractor_queries",
        "attention_dot_products",
        "pretraining_examples",
        "pretraining_optimizer_updates",
        "pretraining_model_queries",
        "model_queries",
        "timing_seconds",
        "timing_is_telemetry_only",
    }
)


def validate_replay_frozen_result(payload: object) -> dict[str, object]:
    """Validate and normalize one permanently nonpromoting result."""
    outer = _object(payload, _OUTER_KEYS, "result")
    if outer["schema"] != RESULT_SCHEMA or outer["comparison_id"] != COMPARISON_ID:
        raise ValueError("result identity does not match the frozen protocol")
    revisions = _strings(outer["paper_revisions"], "paper_revisions")
    if revisions != cast(tuple[str, ...], PROTOCOL["papers"]):
        raise ValueError("paper revisions drift")
    commits = _strings(outer["official_commits"], "official_commits")
    if commits != (REPLAY_OFFICIAL_CODE, RANDUMB_COMMIT, RANPAC_COMMIT, PROL_COMMIT):
        raise ValueError("official commits drift")
    gaps = _strings(outer["protocol_gaps"], "protocol_gaps")
    if gaps != PROTOCOL_GAPS:
        raise ValueError("protocol gaps must remain complete")
    arm = outer["arm"]
    if type(arm) is not str or arm not in _ARMS:
        raise ValueError("unknown replay/frozen arm")
    family, expected_hp = _ARMS[arm]
    if outer["family"] != family:
        raise ValueError("arm family drift")
    seed = _int(outer["seed"], "seed")
    n_tasks = _int(outer["n_tasks"], "n_tasks", 1)
    task_length = _int(outer["task_length"], "task_length", 1)
    input_dim = _int(outer["input_dim"], "input_dim", 1)
    hidden1 = _int(outer["hidden1"], "hidden1", 1)
    hidden2 = _int(outer["hidden2"], "hidden2", 1)
    classes = _int(outer["n_classes"], "n_classes", 2)
    observations = _int(outer["observations"], "observations", 1)
    if n_tasks > _INT32_MAX // task_length or observations != n_tasks * task_length:
        raise ValueError("observation axes drift")
    if _strings(outer["allowed_boundary_information"], "boundary information"):
        raise ValueError("task boundaries are forbidden")
    if _strings(outer["allowed_task_information"], "task information") != (
        "current_example_label",
    ):
        raise ValueError("task information drift")
    hp_object = _object(outer["hyperparameters"], frozenset(expected_hp), "hyperparameters")
    hp = {name: _float(hp_object[name], f"hyperparameters.{name}") for name in expected_hp}
    if hp != expected_hp:
        raise ValueError("hyperparameters drift from the registered arm")
    metric_object = _object(
        outer["metrics"],
        frozenset({"mean_online_accuracy", "mean_loss", "mean_plasticity"}),
        "metrics",
    )
    metrics = {name: _float(metric_object[name], f"metrics.{name}") for name in metric_object}
    if metrics["mean_online_accuracy"] > 1.0 or metrics["mean_plasticity"] > 1.0:
        raise ValueError("bounded metrics must lie in [0,1]")
    resource_object = _object(outer["resources"], _RESOURCE_KEYS, "resources")
    resources = {
        name: _int(resource_object[name], f"resources.{name}")
        for name in _RESOURCE_KEYS
        if name not in {"timing_seconds", "timing_is_telemetry_only"}
    }
    expected_resources = _expected_resources(
        family, observations, input_dim, hidden1, hidden2, classes
    )
    if resources != expected_resources:
        raise ValueError("resource receipt does not match exact derived execution costs")
    timing = _float(resource_object["timing_seconds"], "timing_seconds", 604_800.0)
    if resource_object["timing_is_telemetry_only"] is not True:
        raise ValueError("timing must remain telemetry-only")
    if type(outer["outcome"]) is not str or outer["outcome"] not in {
        "supported",
        "rejected",
        "inconclusive",
    }:
        raise ValueError("invalid retained outcome")
    if outer["negative_outcome_retained"] is not True:
        raise ValueError("negative outcomes must be retained")
    if outer["development_only"] is not True or outer["scientific_promotion_allowed"] is not False:
        raise ValueError("the protocol must remain permanently nonpromoting")
    return {
        **dict(outer),
        "paper_revisions": list(revisions),
        "official_commits": list(commits),
        "protocol_gaps": list(gaps),
        "seed": seed,
        "n_tasks": n_tasks,
        "task_length": task_length,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "n_classes": classes,
        "observations": observations,
        "hyperparameters": hp,
        "metrics": metrics,
        "resources": {**resources, "timing_seconds": timing, "timing_is_telemetry_only": True},
    }


def validate_matched_replay_frozen_results(payloads: object) -> tuple[dict[str, object], ...]:
    if type(payloads) not in (list, tuple) or len(cast(list[object], payloads)) != len(_ARMS):
        raise ValueError("matched results must contain exactly all eight arms")
    results = tuple(validate_replay_frozen_result(item) for item in cast(list[object], payloads))
    if {result["arm"] for result in results} != set(_ARMS):
        raise ValueError("matched results must contain every arm exactly once")
    axes = (
        "seed",
        "n_tasks",
        "task_length",
        "input_dim",
        "hidden1",
        "hidden2",
        "n_classes",
        "observations",
    )
    if any(any(result[name] != results[0][name] for name in axes) for result in results[1:]):
        raise ValueError("matched seed/workload axes drift")
    return results


def expected_resources_for_result(
    family: str, observations: int, input_dim: int, hidden1: int, hidden2: int, classes: int
) -> dict[str, int]:
    """Expose the exact derived resource record to the runner receipt builder."""
    return _expected_resources(family, observations, input_dim, hidden1, hidden2, classes)
