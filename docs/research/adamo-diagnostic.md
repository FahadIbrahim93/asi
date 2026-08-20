# AdamO dynamical-isometry diagnostic

Status: bounded development comparator; permanently nonpromoting; no result is recorded here.

## Source audit

The implementation target is Rosseau, Müller, and Nowé, *Preserving Plasticity in Continual
Learning via Dynamical Isometry*, exactly [arXiv:2606.09762v1](https://arxiv.org/abs/2606.09762v1),
submitted 8 June 2026. The adapter implements the pseudo-orthogonality penalty and the decoupled
AdamO update described by equations 16, 19, and 20. The paper reports orthogonally initialized
depth-4, width-512 MLPs at learning rate `1e-4`, eight seeds, and AdamO penalty strength `1e-3`;
its supervised tasks include random-label CIFAR-10, permuted MNIST/CIFAR-10, and label-shuffled
CIFAR-100. It also reports CNN, continual PPO, and preliminary transformer experiments.

No author-maintained repository or code link was present in the v1 paper, arXiv metadata, or
author/title/arXiv-ID GitHub searches on 17 August 2026. The only indexed AdamO catalog found also
marked official code absent. Consequently `official_code` is null, no commit is invented, and
official parity is fail-closed. A newly published author repository requires a fresh commit pin
and source-level audit before any parity claim.

## Executable comparison

`asi-adamo-diagnostic` runs four matched arms through the existing IPMNIST screening runner:

- live AdamW control;
- `adamo_inert`, whose zero isometry strength must reduce bit-exactly to AdamW;
- AdamO at the paper's `1e-3` penalty strength;
- a causal ablation that mixes task and isometry gradients in Adam's moments rather than applying
  AdamO's decoupled penalty step.

All arms freeze the dataset, initialization root, task permutations, example schedule, seed,
observations, and updates. The learner receives no boundary identifier; it sees only each current
example and label. A post-task observer, downstream of learning, evaluates the end-to-end
input/output Jacobian on fixed dataset row zero and records its singular-value range, clipped
condition number, RMS distance from one, and the layer weight-Gram penalty. It also binds exact
parameter and learner-state hashes. The observer's task index and sentinel never enter an update.

Run only on a caller-materialized NPZ containing exactly float32 `inputs` and int32 `labels`:

```bash
.venv/bin/asi-adamo-diagnostic --catalog
.venv/bin/asi-adamo-diagnostic --dataset /new/path/data.npz \
  --profile contract-smoke --seed 15600
.venv/bin/asi-adamo-matched-development --dataset /new/path/data.npz \
  --profile bounded-development --output /new/path/adamo-matched.json
```

Receipts bind exact dataset and current-source hashes, runtime identity, data steps, observations,
updates, logical model queries, reverse-mode Jacobian rows, persistent numeric bytes, peak Gram
workspace, and a named logical-compute convention. Timing is telemetry only. Successful execution
means only that an uninterpreted development measurement was produced. Every outcome is retained;
the validator permanently rejects promotion fields, unmatched axes, malformed numerics, resource
drift, source drift, and a non-exact inert reduction.

The complete campaign crosses the four arms with consumed development seeds
`15600` through `15604`. It binds the exact supplied dataset, every
Threefry-derived initialization and task/example schedule, the installed
execution-source closure, and the complete Python/JAX/dependency runtime. Its
aggregate records per-arm whole-stream curves, final diagnostic means, and
summed additive work counters plus maximum per-shard memory envelopes. Strict
publication validation reruns all five seed shards under the current source and
compares every deterministic field;
wall-clock timing remains telemetry only. The campaign decision must remain
`inconclusive` because there is no registered decision rule. Its atomic
no-replace writer replays before publication. Consistency hashes are not
authenticated execution attestation.

## Comparability and execution gates

This is an IPMNIST adaptation, not reproduction of any paper result. It uses the current
784-300-150-10 ReLU runner with its existing initialization rather than the paper's orthogonally
initialized depth-4 width-512 MLP, and the small registered profiles are qualification budgets,
not paper horizons. It does not implement the convolution-kernel reshape, GroupSort,
Newton-Schulz, ReLU-revival, empirical NTK, effective-rank, CNN, RL, or transformer protocols.

Before a scientific comparison: locate and pin official code or independently verify equations;
implement the exact paper dataset/task construction, architecture, initialization and full
diagnostic definitions; qualify the missing model families; freeze a separate preregistered
protocol and untouched seeds; and establish calibrated compute, memory, and timing gates. The five
listed seeds are consumed development seeds and can never promote a claim.
