"""Multi-seed experiment runner for publication-quality analysis.

Provides infrastructure for running experiments across multiple seeds
with optional parallelization and aggregation of results.

Two conventions matter for downstream analysis:

* :func:`run_multi_seed_experiment` runs every config on the *same* seed
  list, so per-seed values are aligned across configs.  Keep it that way:
  the significance tests in :mod:`alberta_framework.utils.statistics`
  default to paired tests, which require seed-matched samples.
* "Final value" summaries are the mean over the last 100 steps (or the
  whole trace when shorter), not the last step — see
  :func:`aggregate_metrics`.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import jax.random as jr
import numpy as np
from numpy.typing import NDArray

from alberta_framework.core.learners import (
    LinearLearner,
    metrics_to_dicts,
    run_learning_loop,
)
from alberta_framework.core.types import LearnerState
from alberta_framework.streams.base import ScanStream


class ExperimentConfig(NamedTuple):
    """Configuration for a single experiment.

    Attributes:
        name: Human-readable name for this configuration
        learner_factory: Callable that returns a fresh learner instance
        stream_factory: Callable that returns a fresh stream instance
        num_steps: Number of learning steps to run
    """

    name: str
    learner_factory: Callable[[], LinearLearner]
    stream_factory: Callable[[], ScanStream[Any]]
    num_steps: int


class SingleRunResult(NamedTuple):
    """Result from a single experiment run.

    Attributes:
        config_name: Name of the configuration that was run
        seed: Random seed used for this run
        metrics_history: List of metric dictionaries from each step
        final_state: Final learner state after training
    """

    config_name: str
    seed: int
    metrics_history: list[dict[str, float]]
    final_state: LearnerState


class MetricSummary(NamedTuple):
    """Summary statistics for a single metric.

    Attributes:
        mean: Mean across seeds
        std: Sample standard deviation across seeds (zero for one seed)
        min: Minimum value across seeds
        max: Maximum value across seeds
        n_seeds: Number of seeds
        values: Raw values per seed
    """

    mean: float
    std: float
    min: float
    max: float
    n_seeds: int
    values: NDArray[np.float64]


class AggregatedResults(NamedTuple):
    """Aggregated results across multiple seeds.

    Attributes:
        config_name: Name of the configuration
        seeds: List of seeds used
        metric_arrays: Dict mapping metric name to (n_seeds, n_steps) array
        summary: Dict mapping metric name to MetricSummary (final values)
    """

    config_name: str
    seeds: list[int]
    metric_arrays: dict[str, NDArray[np.float64]]
    summary: dict[str, MetricSummary]


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
) -> SingleRunResult:
    """Run a single experiment with a given seed.

    Args:
        config: Experiment configuration
        seed: Random seed for the stream

    Returns:
        SingleRunResult with metrics and final state
    """
    learner = config.learner_factory()
    stream = config.stream_factory()
    key = jr.key(seed)

    result = run_learning_loop(learner, stream, config.num_steps, key)
    final_state, metrics = cast(tuple[LearnerState, Any], result)
    normalized = learner.normalizer is not None
    metrics_history = metrics_to_dicts(metrics, normalized=normalized)

    return SingleRunResult(
        config_name=config.name,
        seed=seed,
        metrics_history=metrics_history,
        final_state=final_state,
    )


def _require_finite_metric_array(arr: NDArray[np.float64], metric: str) -> None:
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"metric {metric!r} contains non-finite samples")


def aggregate_metrics(results: list[SingleRunResult]) -> AggregatedResults:
    """Aggregate results from multiple seeds into summary statistics.

    Each metric's per-seed "final value" is the mean over the last 100 steps
    (the whole trace when shorter).  The 100-step window is a fixed
    convention here, chosen to estimate settled performance less noisily
    than the last step; use :func:`get_final_performance` when a different
    window is needed.

    Args:
        results: List of SingleRunResult from multiple seeds

    Returns:
        AggregatedResults with aggregated metrics

    Raises:
        ValueError: If ``results`` is empty, or any metric sample is non-finite.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    config_name = results[0].config_name
    seeds = [r.seed for r in results]

    # Get all metric keys from first result
    if any(
        not r.metrics_history
        or any(set(metrics) != set(results[0].metrics_history[0]) for metrics in r.metrics_history)
        for r in results
    ):
        raise ValueError("all runs must contain the same metric keys at every step")
    metric_keys = list(results[0].metrics_history[0].keys())

    # Build metric arrays: (n_seeds, n_steps)
    metric_arrays: dict[str, NDArray[np.float64]] = {}
    for key in metric_keys:
        arrays = []
        for r in results:
            values = np.array([m[key] for m in r.metrics_history])
            arrays.append(values)
        metric_arrays[key] = np.stack(arrays)
        _require_finite_metric_array(metric_arrays[key], key)

    # Compute summary statistics for final values (mean of last 100 steps)
    summary: dict[str, MetricSummary] = {}
    n_seeds = len(results)
    for key in metric_keys:
        # Use mean of last 100 steps as the final value
        window = min(100, metric_arrays[key].shape[1])
        final_values = np.mean(metric_arrays[key][:, -window:], axis=1)
        summary[key] = MetricSummary(
            mean=float(np.mean(final_values)),
            std=float(np.std(final_values, ddof=1)) if n_seeds > 1 else 0.0,
            min=float(np.min(final_values)),
            max=float(np.max(final_values)),
            n_seeds=n_seeds,
            values=final_values,
        )

    return AggregatedResults(
        config_name=config_name,
        seeds=seeds,
        metric_arrays=metric_arrays,
        summary=summary,
    )


def run_multi_seed_experiment(
    configs: Sequence[ExperimentConfig],
    seeds: int | Sequence[int] = 30,
    parallel: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> dict[str, AggregatedResults]:
    """Run experiments across multiple seeds with optional parallelization.

    Args:
        configs: List of experiment configurations to run. Names must be unique.
        seeds: Number of seeds (generates 0..n-1) or explicit list of seeds.
            Explicit seeds must be unique.
        parallel: Whether to use parallel execution (requires joblib)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        Dictionary mapping config name to AggregatedResults

    Raises:
        ValueError: If two or more configurations have the same name, or if the
            explicit seed list contains duplicates
    """
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for config in configs:
        if config.name in seen_names:
            duplicate_names.add(config.name)
        else:
            seen_names.add(config.name)

    if duplicate_names:
        formatted_names = ", ".join(repr(name) for name in sorted(duplicate_names))
        raise ValueError(
            f"Experiment configuration names must be unique; duplicates: {formatted_names}"
        )

    # Convert seeds to list
    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = list(seeds)

    seen_seeds: set[int] = set()
    duplicate_seeds: set[int] = set()
    for seed in seed_list:
        if seed in seen_seeds:
            duplicate_seeds.add(seed)
        else:
            seen_seeds.add(seed)

    if duplicate_seeds:
        formatted_seeds = ", ".join(str(seed) for seed in sorted(duplicate_seeds))
        raise ValueError(f"Experiment seeds must be unique; duplicates: {formatted_seeds}")

    # Build list of (config, seed) pairs
    tasks: list[tuple[ExperimentConfig, int]] = []
    for config in configs:
        for seed in seed_list:
            tasks.append((config, seed))

    # Run experiments
    if parallel:
        try:
            from joblib import Parallel, delayed

            if show_progress:
                try:
                    from tqdm import tqdm

                    results_list: list[SingleRunResult] = Parallel(n_jobs=n_jobs)(
                        delayed(run_single_experiment)(config, seed)
                        for config, seed in tqdm(tasks, desc="Running experiments")
                    )
                except ImportError:
                    results_list = Parallel(n_jobs=n_jobs)(
                        delayed(run_single_experiment)(config, seed) for config, seed in tasks
                    )
            else:
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_experiment)(config, seed) for config, seed in tasks
                )
        except ImportError:
            # Fallback to sequential if joblib not available
            results_list = _run_sequential(tasks, show_progress)
    else:
        results_list = _run_sequential(tasks, show_progress)

    # Group results by config name
    grouped: dict[str, list[SingleRunResult]] = {}
    for result in results_list:
        if result.config_name not in grouped:
            grouped[result.config_name] = []
        grouped[result.config_name].append(result)

    # Aggregate each config
    aggregated: dict[str, AggregatedResults] = {}
    for config_name, group_results in grouped.items():
        aggregated[config_name] = aggregate_metrics(group_results)

    return aggregated


def _run_sequential(
    tasks: list[tuple[ExperimentConfig, int]],
    show_progress: bool,
) -> list[SingleRunResult]:
    """Run experiments sequentially."""
    if show_progress:
        try:
            from tqdm import tqdm

            return [run_single_experiment(config, seed) for config, seed in tqdm(tasks)]
        except ImportError:
            pass
    return [run_single_experiment(config, seed) for config, seed in tasks]


def get_metric_timeseries(
    results: AggregatedResults,
    metric: str = "squared_error",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Get the mean and one-sample-standard-deviation band for a metric.

    Args:
        results: Aggregated results
        metric: Name of the metric

    Returns:
        Tuple of ``(mean, mean - sample_std, mean + sample_std)`` arrays. A
        one-seed aggregate has zero spread and therefore a point band.
    """
    arr = results.metric_arrays[metric]
    _require_finite_metric_array(arr, metric)
    mean = np.mean(arr, axis=0)
    std = np.zeros_like(mean) if arr.shape[0] == 1 else np.std(arr, axis=0, ddof=1)
    return mean, mean - std, mean + std


def get_final_performance(
    results: dict[str, AggregatedResults],
    metric: str = "squared_error",
    window: int = 100,
) -> dict[str, tuple[float, float]]:
    """Get final performance (mean, sample std) for each config.

    Args:
        results: Dictionary of aggregated results
        metric: Metric to evaluate
        window: Number of final steps to average. Must be positive. A window
            longer than the trace uses the whole trace, matching
            :func:`aggregate_metrics`.

    Returns:
        Dictionary mapping config name to ``(mean, sample_std)``. A one-seed
        aggregate reports zero spread.

        Raises:
        ValueError: If ``window`` is not positive, a metric array has no
            time steps, or any metric sample is non-finite. ``window=0`` is
            not "the last step": ``arr[:, -0:]`` is the full trace, so the
            helper refuses rather than silently reporting a full-horizon mean.
    """
    if window <= 0:
        raise ValueError(f"window must be positive (got {window})")

    performance: dict[str, tuple[float, float]] = {}
    for name, agg in results.items():
        arr = agg.metric_arrays[metric]
        if arr.shape[1] == 0:
            raise ValueError(
                f"AggregatedResults {name!r} must contain at least one metric step "
                f"for {metric!r}"
            )
        _require_finite_metric_array(arr, metric)
        final_window = min(window, arr.shape[1])
        final_means = np.mean(arr[:, -final_window:], axis=1)
        std = float(np.std(final_means, ddof=1)) if len(final_means) > 1 else 0.0
        performance[name] = (float(np.mean(final_means)), std)
    return performance


def extract_hyperparameter_results(
    results: dict[str, AggregatedResults],
    metric: str = "squared_error",
    param_extractor: Callable[[str], Any] | None = None,
) -> dict[Any, tuple[float, float]]:
    """Extract results indexed by hyperparameter value.

    Useful for creating hyperparameter sensitivity plots.

    Args:
        results: Dictionary of aggregated results
        metric: Metric to evaluate
        param_extractor: Function to extract param value from config name

    Returns:
        Dictionary mapping param value to (mean, std) tuple
    """
    performance = get_final_performance(results, metric)

    if param_extractor is None:
        return {k: v for k, v in performance.items()}

    return {param_extractor(name): perf for name, perf in performance.items()}
