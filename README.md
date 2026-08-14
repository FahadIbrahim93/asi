# Alberta Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)

Alberta Framework is a JAX research package for continual learning and continual
reinforcement learning, guided by
[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173). It provides online
learners, adaptive optimizers, prediction and control agents, learned-state mechanisms,
planning and option components, non-stationary streams, benchmarks, and strict evidence
validators.

This repository is a development fork of `lalalune/alberta` at fork point `2ac3533`.
[VENDORING.md](VENDORING.md) records the relationship to upstream and the local divergence.
The continual-RL subset is also imported in-process by the elizaOS robot track, so the
package keeps Python 3.12 compatibility and a NumPy 1.26 floor.

## Research status

The framework contains implementation surfaces related to all twelve steps of the Alberta
Plan. That is not an integrated Alberta Plan completion claim. Individual mechanisms range
from contract-tested kernels to narrow historical evidence packages; important end-to-end,
retention, control-benefit, resource-matching, and integration gates remain open.

Keep these boundaries in mind:

- Development and screening runs are permanently nonpromoting unless a separate frozen
  protocol explicitly says otherwise.
- A passing unit test, smoke run, replay, or benchmark does not promote a scientific claim.
- Registered evidence claims are narrow. Acceptance of one does not certify the package or
  establish Alberta Plan completion.
- Pinned artifacts are immutable historical records. Source drift makes compatibility checks
  fail closed; it is not repaired by editing the artifact or loosening its validator.
- Consumed development or evidence seeds cannot be reused as fresh promotion seeds.

See [the research status](docs/status.md) for the current requirement-to-evidence map and
[the evidence methodology](docs/evidence/methodology.md) for promotion rules, artifact
contracts, and validator semantics.

## Install

Alberta Framework requires Python 3.12 or newer.

```bash
pip install alberta-framework
```

Optional dependency groups are available for common workflows:

```bash
pip install 'alberta-framework[gymnasium]'  # Gymnasium adapters
pip install 'alberta-framework[forager]'    # continual-foragax testbed
pip install 'alberta-framework[gpu]'        # JAX CUDA 12 build
pip install 'alberta-framework[dev]'        # tests, lint, and type checking
```

For repository development, use the project virtual environment for every command:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Quick start

This example runs an online linear predictor on a drifting synthetic stream. JAX keys are
explicit, and the learning loop uses `jax.lax.scan`.

```python
import jax.random as jr

from alberta_framework import (
    Autostep,
    LinearLearner,
    RandomWalkStream,
    run_learning_loop,
)

stream = RandomWalkStream(feature_dim=10, drift_rate=0.01)
learner = LinearLearner(optimizer=Autostep())

state, metrics = run_learning_loop(
    learner,
    stream,
    num_steps=10_000,
    key=jr.key(42),
)
```

The repository also exposes short Step 1 and Step 2 integration probes:

```bash
.venv/bin/alberta-step1-smoke --steps 256 --seed 0
.venv/bin/alberta-step2-smoke --steps 128 --seed 0
```

These commands check that the selected kernel runs and returns finite metrics. They are not
scientific experiments or evidence gates.

## Package layout

```text
alberta_framework/
  core/         online learners, optimizers, control, state, models, memory,
                planning, options, feature lifecycles, and agent composition
  streams/      synthetic prediction, closed-loop, Pavlovian, and recurring
                multi-agent streams
  evaluation/   evidence schemas, strict validators, registries, and CLIs
  benchmarks/   IPMNIST and Forager integrations and campaign runners
  utils/        experiment, metric, statistics, and export helpers
  steps/        public Step 1-12 mechanism kernels and smoke integration
tests/          unit, integration, scientific, and replay tests
outputs/        evidence and campaign artifacts; see the immutability rules
```

Most numerical state is represented by immutable Chex dataclasses and carried as JAX PyTrees.
Randomness is passed explicitly. Host orchestration, artifact validation, external benchmark
loading, and some bounded lifecycle operations remain Python-level by design.

The major package surfaces include:

| Area | Examples |
|---|---|
| Online prediction | `LinearLearner`, `MLPLearner`, TD learners, Horde |
| Adaptation | LMS, IDBD, Autostep, SwiftTD, UPGD, normalization, bounding |
| Control | SARSA, actor-critic, average-reward and off-policy variants |
| Continual mechanisms | learned state, feature lifecycles, memory, world models |
| Temporal abstraction | subtasks, STOMP, OaK, option models and bounded planning |
| Composition | `PrototypeAgent` and explicit transition/decision ownership |
| Evaluation | versioned artifacts, strict validators, evidence registry |

API presence means that a mechanism is available for research. It does not imply empirical
benefit, calibrated thresholds, autonomous integration, or scientific acceptance.

## Evidence registry

From a repository checkout, inspect every registered claim with:

```bash
.venv/bin/alberta-evidence-status
```

The exit-code contract is:

| Code | Meaning |
|---:|---|
| `0` | every registered narrow claim is accepted |
| `1` | at least one artifact is missing or is a valid rejection |
| `2` | at least one artifact is invalid |

The registry validates artifact schema, protocol metadata, and registered source hashes. It is
an operational index of narrow claims, not a package-wide evidence score or completion
certificate.

Wheels and source distributions intentionally exclude `outputs/`. Consequently, running the
status command from a normal package installation reports missing artifacts. Use a checkout
when validating the repository's stored evidence chain.

Do not overwrite, repair, or regenerate a pinned artifact in place. A new run must use a new
path and, when required by its contract, a new schema version. The full rules are in
[the evidence methodology](docs/evidence/methodology.md).

## Active development campaigns

The current headline lane is IPMNIST screening and confirmation. It is development-grade and
permanently nonpromoting. Results change as new shards are appended, so this README does not
copy arm rankings, means, test counts, or seed counts.

Use the primary records instead:

- [IPMNIST theory and forward hypotheses](docs/research/ipmnist-theory.md)
- [Campaign runbook](outputs/ipmnist_screening/RUNBOOK.md)
- [Stored development report](outputs/ipmnist_screening/FINAL_REPORT.md)
- [Artifact and reproducibility audit](outputs/ipmnist_screening/AUDIT.md)
- [Publication-run record](outputs/ipmnist_screening/publication_runs/RESULTS.md)
- `outputs/ipmnist_screening/summary_*.json` for the latest stored summaries

Remeasure the intended baseline under the current development protocol before making an A/B
comparison. Do not infer a scientific or state-of-the-art claim from the screening record.

Forager integration and comparator details are in
[FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md).

Before repeating a failed or bounded idea, check
[the negative-results ledger](docs/evidence/negative-results.md).

## Development and testing

Run targeted tests first, then broaden verification as appropriate:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
```

The repository uses these pytest markers:

- `unit`: fast, isolated behavior or contract tests
- `integration`: component, persistence, process, or CLI boundaries
- `scientific`: frozen promoted-evidence protocols
- `slow`: wall-clock-heavy tests excluded from the fast per-change lane

Benchmark campaigns run through their scripts or console CLIs, never as ordinary pytest work.
Keep tests CI-cheap unless the protocol is deliberately registered as scientific evidence.

Library changes should start with a failing test. Preserve immutable state, explicit
`jax.random` keys, Python 3.12 support, and the NumPy 1.26 minimum. Before editing evaluation
or benchmark sources, check whether a stored artifact registers their hashes.

Do not auto-promote results, retune a frozen threshold after seeing held-out data, reuse
consumed seeds, or modify immutable `outputs/` records. See
[the evidence methodology](docs/evidence/methodology.md) before changing any evidence lane.

## Documentation

### Status and evidence

- [Research status and completion gates](docs/status.md)
- [Evidence methodology and property map](docs/evidence/methodology.md)
- [Negative and bounded results](docs/evidence/negative-results.md)

### Runbooks

- [Foragax open development screen](docs/runbooks/foragax-open-screen.md)

### Research and historical audits

- [IPMNIST theory](docs/research/ipmnist-theory.md)
- [RTU Taylor-correction derivation](docs/design/rtu-taylor-correction.md)
- [Forager comparator audit](docs/archive/forager-comparator-audit.md)
- [Historical Forager reconstruction](docs/archive/historical-forager-reconstruction.md)
- [OPMNIST closure provenance](outputs/step2_canonical/step2_opmnist_solution_800task_3seed_PROVENANCE.md)

### Repository and benchmark records

- [Forager benchmark](FORAGER_BENCHMARK.md)
- [Vendoring and fork history](VENDORING.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)

## Citation

Project citation metadata is provided in [CITATION.cff](CITATION.cff). Cite the original papers
for algorithms and benchmarks used in a particular experiment, including the
[Alberta Plan](https://arxiv.org/abs/2208.11173).

## License

Alberta Framework is licensed under the [Apache License 2.0](LICENSE).
