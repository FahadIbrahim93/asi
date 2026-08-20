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
```

Receipts bind exact dataset and current-source hashes, runtime identity, data steps, observations,
updates, logical model queries, reverse-mode Jacobian rows, persistent numeric bytes, peak Gram
workspace, and a named logical-compute convention. Timing is telemetry only. Successful execution
means only that an uninterpreted development measurement was produced. Every outcome is retained;
the validator permanently rejects promotion fields, unmatched axes, malformed numerics, resource
drift, source drift, and a non-exact inert reduction.

## Comparability and execution gates

This is an IPMNIST adaptation, not reproduction of any paper result. It uses the current
784-300-150-10 ReLU runner with its existing initialization rather than the paper's orthogonally
initialized depth-4 width-512 MLP, and the small registered profiles are qualification budgets,
not paper horizons. It does not implement the convolution-kernel reshape, GroupSort,
Newton-Schulz, ReLU-revival, empirical NTK, effective-rank, CNN, RL, or transformer protocols.

Before a scientific comparison: locate and pin official code or independently verify equations;
implement the exact paper dataset/task construction, architecture, initialization and full
diagnostic definitions; qualify the missing model families; freeze a separate preregistered
protocol and untouched seeds; and establish calibrated compute, memory, and timing gates. The four
diagnostic seeds `15600`--`15603` are consumed development seeds and can never promote a claim.

## Frozen matched campaign

`asi-adamo-matched-development` is the only campaign entry point. It freezes fresh seeds
`15610`--`15613`, the `bounded-development` profile, all four arms, canonical OpenML
`mnist_784` version 1 rows 0--59999, a paired two-sided Student-t summary, and one immutable
output at `outputs/adamo_matched_development/report.v1.json`. The command reserves that path
before loading data or training and refuses overwrite, symlink traversal, incomplete seed/arm
matrices, source/runtime/data drift, malformed receipts, reconstructed-statistic drift, and any
promotion or parity field.

```bash
.venv/bin/asi-adamo-matched-development --data-home /path/to/openml-cache
```

This is a small IPMNIST adaptation: eight tasks of 64 updates on the existing
784-300-150-10 ReLU MLP. It is neither a paper reproduction nor evidence. The report retains
every seed receipt, per-arm steps/query/memory/compute/timing accounting, post-task
Jacobian/isometry diagnostics, paired primary intervals, and negative or inconclusive outcomes.
Timing remains telemetry and no campaign outcome may promote a claim. Consistency hashes are not
authenticated execution attestation.
