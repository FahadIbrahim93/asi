# TeLAPA policy-archive qualification smoke

This lane exercises ASI's byte-bounded `BoundedPolicyArchive` in an executable
current `SwitchingTwoStateMDP` stream. It is a permanently nonpromoting
development smoke, not TeLAPA, a paper reproduction, a benchmark result, or a
state-of-the-art claim.

## Provenance audit

The cited paper is *Beyond Single-Model Optimization: Preserving Plasticity in
Continual Reinforcement Learning*, pinned to `arXiv:2604.15414v1` (submitted
2026-04-16). The paper discloses an anonymous code view at
`https://anonymous.4open.science/r/telapa-map_elites-54E8/`. That view does not
expose a reviewable immutable Git commit or content-tree digest in the paper,
and the restricted audit environment could not retrieve it. The catalog
therefore records no code revision or tree identity and categorically rejects
paper parity. A mutable anonymous URL must never be substituted for a commit.

The paper's mechanism is substantially larger than this lane: per-task policy
neighborhoods, PPO, post-training MAP-Elites illumination, few-shot origin
selection, and a learned trajectory embedding maintained with anchors, replay,
alignment, and periodic archive re-embedding. It evaluates recurring MiniGrid
tasks and reports standardized time-to-threshold, success, backward transfer,
and transfer diagnostics. Its appendix also studies same-morphology MuJoCo
friction shifts. None of those protocols or metrics are implemented here.

## Development protocol

Each frozen development seed runs four arms for the same bounded number of
environment steps, observations, online table updates, action queries,
descriptor queries, and boundary disclosures:

- `diverse_archive`: distance-filtered immutable policy snapshots;
- `one_model`: the latest snapshot only;
- `fixed_snapshot`: the first boundary snapshot only;
- `mechanism_off`: bypasses the archive and retains the same fixed anchor.

The live adapter uses the repository's action-dependent two-state switching
environment. A policy is a 2x2 float32 action-value table. At each declared
phase boundary, a JIT-compatible descriptor summarizes state occupancy, action
occupancy, reward, and action switching. The descriptor has no learned state.
Every arm creates an explicit `threefry2x32` root, and the workload identity
and receipt bind the implementation, root count, derivation count, and root
bytes; the ambient JAX PRNG default cannot select this protocol's generator.
Task boundaries are visible only to archive maintenance; task identity is not
an input to the policy, future tasks are hidden, and prior environments cannot
be queried.

The fixed-snapshot and archive-off paths must have identical observation,
action, reward, initial-policy, and final-policy hashes. Resource receipts
separately count environment steps, observations, updates, policy/descriptor/
archive-selection and anchor-selection queries, disclosed boundaries, payload
bytes, active policy bytes, archive or anchor bytes, RNG roots/derivations/root
bytes, and environment-state bytes. The result reports exact per-arm reward
sums and means plus descriptive
per-seed and mean deltas for `diverse_archive` against both requested controls.
Those comparisons are nonpromoting diagnostics, not a selection or benefit
claim. Timing is absent and telemetry-only. Every valid development outcome,
including a tie or regression, can be retained through the create-only Python
publisher at
`outputs/telapa/local-comparator.v2/result.<content-sha256>.json`. The CLI itself
does not write `outputs/`. Publication remains unauthorized until this
prospective protocol is independently reviewed and merged.

The non-catalog CLI, public comparator, and publisher are hard-disabled behind
a separate authorization transition. Contract tests use only the private
bounded executor. Publication reserves its content name through pinned
no-follow directory descriptors before deterministic replay validation, then
uses a no-replace link, fsync, bounded reread, and strict reload validation.

Run the CI-cheap smoke or print only its catalog:

```bash
.venv/bin/asi-telapa-qualification-smoke --steps 32 --phase-length 4
.venv/bin/asi-telapa-qualification-smoke --catalog
```

The CLI emits schema `asi.telapa_qualification_smoke.development.v2`; its
strict validator rejects the earlier smoke schema rather than upgrading it.

## Gates still closed

- establish and license-review an immutable official source revision and tree;
- reproduce the exact dependency/runtime lock and source/config identities;
- implement the paper environments, curricula, PPO and MAP-Elites budgets;
- implement the learned embedder, normalization, anchors, replay, alignment,
  re-embedding, trajectory banks, and few-shot selection;
- match all environment, evaluation, optimizer-reset, task-information, seed,
  query, compute, peak-memory, and persistent-state contracts;
- run strong paper baselines and causal archive/descriptor/maintenance
  ablations, then an untouched preregistered scientific evaluation.

Until all gates close, the lane cannot support TeLAPA parity, performance,
transfer, continual-learning benefit, robotics readiness, or SOTA claims.
