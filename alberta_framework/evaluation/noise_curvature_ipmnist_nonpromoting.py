"""Fail-closed receipts for the noise-curvature IPMNIST development lane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    PAPER_REVISION,
    noise_curvature_persistent_bytes,
)

RESULT_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.development-result.v1"
COMPARISON_ID: Final[str] = "asi.noise-curvature-ipmnist.current-runner.v1"
OFFICIAL_CODE_STATUS: Final[str] = "none-identified-as-of-2026-08-17"
LIVE_CONTROL: Final[str] = "rls_head_resid_l1_preset005"
DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

DEVELOPMENT_GATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "integrity": (
            "every shard validates, the runner admits no diagnostic failure, and all shards "
            "bind one current source/dataset/runtime identity"
        ),
        "mechanism": (
            "the paired 95% interval for joint minus fixed-Adam+L2 mean online accuracy "
            "lies strictly above zero"
        ),
        "causal": (
            "paired 95% intervals for joint minus each single-signal ablation lie strictly "
            "above zero"
        ),
        "hillclimb": (
            "joint minus live-control mean online accuracy is at least +0.005 and its paired "
            "95% interval lies strictly above zero"
        ),
        "transfer": (
            "a screen survivor must be remeasured on the 200-task development confirmation "
            "and recurrence/control lanes before reference-dev consideration"
        ),
        "resources": (
            "persistent bytes and query counts must validate exactly; timing remains telemetry "
            "until separately qualified"
        ),
    }
)

PROTOCOL_DIFFERENCES: Final[tuple[str, ...]] = (
    "ASI uses input-permuted MNIST labels, not the paper's random-label MNIST tasks",
    "ASI uses its current 784-300-150-10 ReLU MLP, not the paper's 784-256-256-10 MLP",
    "ASI performs one online update per example, not batch-256 training for 250 epochs per task",
    "the most recent K examples form the charged diagnostic minibatch; optimizer "
    "updates stay online",
    "ASI uses decoupled AdamW weight decay for the fixed L2 conditioning arm",
    "one retained block-power direction and one absolute block-Rayleigh HVP per layer are "
    "used at each K-step decision",
    "each block-power direction starts as a deterministic equal-weight vector because arXiv "
    "v3 does not specify its initializer",
    "curvature statistics use bias-corrected EMA mean/variance without the paper's finite queue",
    "kappa is fixed to 1 because arXiv v3 leaves its absorbed scaling constants unspecified",
    "the paper's much-less-than warm condition is operationalized as 0.1 times the safe bound",
    "timing remains telemetry-only and no empirical result is represented by this implementation",
)

_ARM_MODES: Final[Mapping[str, float]] = MappingProxyType(
    {
        "noise_curvature_fixed_adam_l2": 0.0,
        "noise_curvature_gradient_only": 1.0,
        "noise_curvature_volatility_only": 2.0,
        "noise_curvature_combined": 3.0,
    }
)

BASE_HYPERPARAMETERS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "step_size": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-3,
        "control_interval": 40.0,
        "power_iterations": 1.0,
        "ema_decay": 0.9,
        "volatility_epsilon": 1e-8,
        "volatility_inflation": 1.0,
        "volatility_kappa": 1.0,
        "safety_factor": 0.8,
        "cool_rate": 0.99,
        "warm_rate": 1.01,
        "warm_fraction": 0.3,
        "timid_fraction": 0.1,
        "effective_step_floor": 0.12,
    }
)
_HP_KEYS = frozenset((*BASE_HYPERPARAMETERS, "controller_mode"))
_INT32_MAX = (1 << 31) - 1
_MAX_STRINGS = 16
_MAX_STRING_BYTES = 512


def registered_hyperparameters(arm: str) -> dict[str, float]:
    """Return an independent exact hyperparameter object for one arm."""

    if type(arm) is not str or arm not in _ARM_MODES:
        raise ValueError(f"arm must be one of {sorted(_ARM_MODES)}")
    return {**BASE_HYPERPARAMETERS, "controller_mode": _ARM_MODES[arm]}


def registered_arms() -> tuple[str, ...]:
    """Return the registered scheduler arms in causal-comparison order."""

    return tuple(_ARM_MODES)


def matched_arm_names() -> tuple[str, ...]:
    """Include the moving campaign leader as the required live strong control."""

    return (LIVE_CONTROL, *registered_arms())


def _object(value: object, keys: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    if set(value) != keys:
        raise ValueError(f"{context} keys must be exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _int(value: object, *, context: str, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value <= _INT32_MAX:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{context} must be an exact {qualifier} signed-int32 integer")
    return value


def _float(value: object, *, context: str, nonnegative: bool = False, unit: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    if unit and not 0.0 <= value <= 1.0:
        raise ValueError(f"{context} must lie in [0,1]")
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) > _MAX_STRINGS:
        raise ValueError(f"{context} must be a bounded list of nonempty UTF-8 strings")
    for item in value:
        if type(item) is not str or not item or "\x00" in item:
            raise ValueError(f"{context} must be a bounded list of nonempty UTF-8 strings")
        try:
            encoded = item.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{context} strings must be valid UTF-8") from error
        if len(encoded) > _MAX_STRING_BYTES:
            raise ValueError(f"{context} must be a bounded list of nonempty UTF-8 strings")
    resolved = tuple(value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{context} must not contain duplicates")
    return resolved


def _checked_product(context: str, left: int, right: int) -> int:
    if left > _INT32_MAX // right:
        raise ValueError(f"{context} exceeds signed int32")
    return left * right


def validate_noise_curvature_development_result(payload: object) -> dict[str, object]:
    """Return a normalized copy after strict identity and accounting checks."""

    outer = _object(
        payload,
        frozenset(
            {
                "schema",
                "comparison_id",
                "paper_revision",
                "official_code_status",
                "protocol_differences",
                "live_control",
                "arm",
                "seed",
                "development_seed_protocol",
                "n_tasks",
                "task_length",
                "input_dim",
                "hidden1",
                "hidden2",
                "n_classes",
                "observations",
                "updates",
                "allowed_boundary_information",
                "allowed_task_information",
                "hyperparameters",
                "metrics",
                "resources",
                "outcome",
                "outcome_retained",
                "development_only",
                "scientific_promotion_allowed",
            }
        ),
        context="result",
    )
    identities = {
        "schema": RESULT_SCHEMA,
        "comparison_id": COMPARISON_ID,
        "paper_revision": PAPER_REVISION,
        "official_code_status": OFFICIAL_CODE_STATUS,
        "live_control": LIVE_CONTROL,
    }
    for key, expected in identities.items():
        if type(outer[key]) is not str or outer[key] != expected:
            raise ValueError(f"{key} does not match the registered protocol")
    differences = _strings(outer["protocol_differences"], context="protocol_differences")
    if differences != PROTOCOL_DIFFERENCES:
        raise ValueError("protocol_differences do not match the registered adaptation")
    arm = outer["arm"]
    if type(arm) is not str or arm not in _ARM_MODES:
        raise ValueError(f"arm must be one of {sorted(_ARM_MODES)}")

    seed = _int(outer["seed"], context="seed")
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed must belong to the frozen development seed set")
    if outer["development_seed_protocol"] != list(DEVELOPMENT_SEEDS):
        raise ValueError("development_seed_protocol must match the frozen seed set")
    n_tasks = _int(outer["n_tasks"], context="n_tasks", positive=True)
    task_length = _int(outer["task_length"], context="task_length", positive=True)
    input_dim = _int(outer["input_dim"], context="input_dim", positive=True)
    hidden1 = _int(outer["hidden1"], context="hidden1", positive=True)
    hidden2 = _int(outer["hidden2"], context="hidden2", positive=True)
    n_classes = _int(outer["n_classes"], context="n_classes", positive=True)
    observations = _int(outer["observations"], context="observations", positive=True)
    updates = _int(outer["updates"], context="updates", positive=True)
    expected_observations = _checked_product("n_tasks * task_length", n_tasks, task_length)
    if observations != expected_observations or updates != observations:
        raise ValueError("observations and updates must equal n_tasks * task_length")
    boundary = _strings(
        outer["allowed_boundary_information"], context="allowed_boundary_information"
    )
    task_info = _strings(outer["allowed_task_information"], context="allowed_task_information")
    if boundary or task_info != ("current_example_label",):
        raise ValueError("allowed boundary/task information does not match the online protocol")

    hp = _object(outer["hyperparameters"], _HP_KEYS, context="hyperparameters")
    normalized_hp = {
        key: _float(hp[key], context=f"hyperparameters.{key}", nonnegative=True) for key in _HP_KEYS
    }
    if normalized_hp != registered_hyperparameters(arm):
        raise ValueError("hyperparameters do not match the registered scheduler arm")
    interval = int(normalized_hp["control_interval"])
    power_iterations = int(normalized_hp["power_iterations"])
    if task_length % interval:
        raise ValueError("task_length must be divisible by control_interval")

    metrics = _object(
        outer["metrics"],
        frozenset({"mean_online_accuracy", "mean_loss", "mean_plasticity"}),
        context="metrics",
    )
    normalized_metrics = {
        "mean_online_accuracy": _float(
            metrics["mean_online_accuracy"], context="mean_online_accuracy", unit=True
        ),
        "mean_loss": _float(metrics["mean_loss"], context="mean_loss", nonnegative=True),
        "mean_plasticity": _float(metrics["mean_plasticity"], context="mean_plasticity", unit=True),
    }

    resources = _object(
        outer["resources"],
        frozenset(
            {
                "persistent_bytes",
                "environment_steps",
                "data_steps",
                "model_queries",
                "first_order_gradient_queries",
                "loss_only_queries",
                "hessian_vector_product_queries",
                "controller_events",
                "timing_seconds",
                "timing_is_telemetry_only",
            }
        ),
        context="resources",
    )
    persistent_bytes = _int(resources["persistent_bytes"], context="persistent_bytes")
    environment_steps = _int(resources["environment_steps"], context="environment_steps")
    data_steps = _int(resources["data_steps"], context="data_steps")
    model_queries = _int(resources["model_queries"], context="model_queries")
    gradient_queries = _int(
        resources["first_order_gradient_queries"], context="first_order_gradient_queries"
    )
    loss_queries = _int(resources["loss_only_queries"], context="loss_only_queries")
    hvp_queries = _int(
        resources["hessian_vector_product_queries"],
        context="hessian_vector_product_queries",
    )
    controller_events = _int(resources["controller_events"], context="controller_events")
    timing = _float(resources["timing_seconds"], context="timing_seconds", nonnegative=True)
    if timing > 604_800.0:
        raise ValueError("timing_seconds exceeds the seven-day development bound")
    if resources["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain true")
    expected_events = observations // interval
    expected_gradient_queries = observations + expected_events * interval
    expected_loss_queries = observations
    expected_hvp_queries = expected_events * 3 * power_iterations
    expected_model_queries = (
        expected_gradient_queries + expected_loss_queries + expected_hvp_queries
    )
    if expected_model_queries > _INT32_MAX:
        raise ValueError("derived model query count exceeds signed int32")
    if (
        environment_steps != 0
        or data_steps != observations
        or controller_events != expected_events
        or gradient_queries != expected_gradient_queries
        or loss_queries != expected_loss_queries
        or hvp_queries != expected_hvp_queries
        or model_queries != expected_model_queries
    ):
        raise ValueError("resource counters do not match executed scheduler semantics")
    parameter_count = (
        _checked_product("input_dim * hidden1", input_dim, hidden1)
        + hidden1
        + _checked_product("hidden1 * hidden2", hidden1, hidden2)
        + hidden2
        + _checked_product("hidden2 * n_classes", hidden2, n_classes)
        + n_classes
    )
    expected_bytes = noise_curvature_persistent_bytes(
        parameter_count=parameter_count,
        input_dim=input_dim,
        control_interval=interval,
    )
    if persistent_bytes != expected_bytes:
        raise ValueError("persistent_bytes does not match the canonical numeric payload")

    if type(outer["outcome"]) is not str or outer["outcome"] not in (
        "supported",
        "rejected",
        "inconclusive",
    ):
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if outer["outcome_retained"] is not True:
        raise ValueError("outcome_retained must permanently remain true")
    if outer["development_only"] is not True:
        raise ValueError("development_only must permanently remain true")
    if outer["scientific_promotion_allowed"] is not False:
        raise ValueError("scientific_promotion_allowed must permanently remain false")

    return {
        **dict(outer),
        "seed": seed,
        "n_tasks": n_tasks,
        "task_length": task_length,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "n_classes": n_classes,
        "observations": observations,
        "updates": updates,
        "metrics": normalized_metrics,
        "resources": dict(resources),
    }


def validate_matched_noise_curvature_results(payloads: object) -> list[dict[str, object]]:
    """Validate one complete same-seed causal scheduler panel.

    The moving RLS incumbent has its own campaign receipt schema and is checked
    separately when a development report is assembled.
    """

    if type(payloads) is not list or len(payloads) != len(_ARM_MODES):
        raise ValueError("matched scheduler panel must contain every arm exactly once")
    validated = [validate_noise_curvature_development_result(item) for item in payloads]
    by_arm = {cast(str, item["arm"]): item for item in validated}
    if len(by_arm) != len(validated) or set(by_arm) != set(_ARM_MODES):
        raise ValueError("matched scheduler panel must contain every arm exactly once")
    first = validated[0]
    for item in validated[1:]:
        for field in (
            "seed",
            "development_seed_protocol",
            "n_tasks",
            "task_length",
            "input_dim",
            "hidden1",
            "hidden2",
            "n_classes",
            "observations",
            "updates",
            "allowed_boundary_information",
            "allowed_task_information",
        ):
            if item[field] != first[field]:
                raise ValueError(f"matched scheduler panel differs on {field}")
        resources = cast(dict[str, object], item["resources"])
        first_resources = cast(dict[str, object], first["resources"])
        for field in (
            "persistent_bytes",
            "environment_steps",
            "data_steps",
            "model_queries",
            "first_order_gradient_queries",
            "loss_only_queries",
            "hessian_vector_product_queries",
            "controller_events",
        ):
            if resources[field] != first_resources[field]:
                raise ValueError(f"matched scheduler panel differs on resource axis {field}")
    return validated
