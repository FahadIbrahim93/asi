# Fork status: Alberta Framework

This directory began as a vendored copy of the **Alberta Framework** — a JAX
implementation of
[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) (Sutton,
Bowling, Pilarski 2022). It is now a **development fork**, not a
lightly-patched vendor drop: the continual-learning research campaign happens
in this tree, and the divergence from the imported snapshot is substantial and
intentional.

- **Fork point:** `lalalune/alberta` @
  `2ac35333efae45cf969ce02ec1f2703476fed6c2`
- **Canonical repository URL:** https://github.com/lalalune/alberta
  (this is the single upstream identity; the `j-klawson/alberta-framework`
  URLs that older `pyproject.toml`/`CITATION.cff` revisions pointed at are
  stale and are no longer referenced)
- **License:** Apache-2.0 (see `LICENSE`)

## Why it lives here

`eliza-robot` (`packages/research/robot`) uses the Alberta continual-RL
control subset to train robot policies that learn a sequence of tasks without
catastrophic forgetting, and benchmarks it against standard RL (PPO). The
framework is imported in-process from the robot's Python 3.12 environment,
which is why `requires-python` is `>=3.12` and the NumPy floor is `>=1.26`
(brax/mujoco pin `numpy<2` there).

## Divergence from the fork point

The fork-point commit is not present in this repository's history (the tree
arrived already diverged), so the divergence is described by capabilities
rather than brittle file and test counts:

- **`alberta_framework/evaluation/`** — fork-local subpackage containing
  strict evidence artifacts and validators, the evidence-registry manifest
  (`evidence_manifest.py` / `alberta-evidence-status`), and the evidence
  CLIs.
- **`alberta_framework/benchmarks/`** — fork-local subpackage containing
  the Forager family (matched-current campaign machinery, RNG parity,
  `official_foragax`/OCI, open screen, historical reconstruction), the
  published-protocol replication lanes (`upgd_ipmnist`,
  `upgd_label_emnist`, `ipmnist_screening`).
- **`alberta_framework/core/`** — additions since the fork point include
  `swift_td`, `stacked_horde`, learned-state and memory components, UPGD,
  option/value-duration support, world models, feature lifecycles,
  and the `PrototypeAgent` composition surface.
- **`alberta_framework/streams/`** — fork-local additions include
  `gauntlet`, `closed_loop`, and `recurring_multiagent`.
- **`tests/`** — tests for upstream-only `benchmarks/`, `examples/`, and
  narrative documents are not carried when their implementation is absent.
- **Top level**: `docs/status.md`, `docs/evidence/methodology.md`,
  `FORAGER_BENCHMARK.md`, the execution runbooks and campaign audits, the
  `outputs/` evidence artifacts, and this file are fork-local.
  `CHANGELOG.md` continues upstream numbering (0.27.0 was cut here), and
  `pyproject.toml` registers the current console scripts.

Because of this, "re-sync from upstream" is no longer a patch-reapplication
exercise; treat any future sync as a merge between diverged development lines.

Not carried from upstream: its repository metadata and non-runtime trees such
as the root-level `benchmarks/` tree, historical `docs/`, `examples/`, and
scripts. This fork has its own `.github/` and `docs/` contents.

## The benchmarks-shim hazard (fixed in 0.27.0)

Upstream kept its benchmark drivers in a repository-root `benchmarks/` tree,
and `alberta_framework/__init__.py` ended with a compatibility shim that
registered that root package under the `alberta_framework.benchmarks` name.
Once this fork added a real `alberta_framework.benchmarks` subpackage, the
shim became a hazard: with any unrelated top-level `benchmarks/` directory
importable (for example an upstream checkout on `sys.path`), the shim could
bind the foreign package into `sys.modules` under the subpackage's name and
shadow the packaged integrations.

As of 0.27.0 the shim is removed. Normal Python submodule resolution loads the
packaged `alberta_framework.benchmarks` tree without importing the benchmark
stack during every base-package import.

## Continual-RL subset used by the robot package

The robot tree's direct imports are `alberta_framework.core.actor_critic`,
`core.continual_backprop`, `core.initializers`, `core.normalizers`,
`core.optimizers`, and the top-level re-exports `SARSAAgent`, `SARSAConfig`,
and `ObGDBounding`. The top-level package must remain importable from the
robot environment, but benchmark campaigns and the 12-step / `diffeml_*` /
prototype machinery are not robot dependencies.
