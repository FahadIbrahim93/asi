# Bounded growing and elastic networks comparison (#1562)

This is a permanently nonpromoting development adaptation of
arXiv `2608.01475v1`. It does not contain a result, reproduce the paper, or
establish state of the art. `asi-bounded-elastic-ipmnist` runs the complete
frozen 5-seed × 4-arm roster and publishes only to a new result path.

## Registered comparison

The current IPMNIST runner compares a preallocated masked structure-off
control, bounded growth, fixed-active-size elastic prune/regrow, and the live
fixed-capacity CBP control. Every arm receives the same seed-derived
initialization, permutation schedule, example schedule, current example label,
and known fixed 5,000-example boundary. All use the same full allocation and
peak persistent-memory ceiling; receipts separately report active parameters,
growth, pruning, structure events, updates, queries, and learner-owned numeric
bytes. Timing is telemetry only.

This adaptation keeps JAX shapes static, begins with half of hidden layer 1
active, and changes structure only at the known boundary. It uses prior-task
online activation sums rather than the paper's future sample, freshly
initializes a slot before reactivation, and uses ASI's two-hidden-layer MLP,
5,000-example task, `[-1,1]` input, and current-runner SGD/CBP conventions. No
official repository was linked by the audited paper revision, so the paper is
the implementation source. These differences prevent paper-result comparison.

## Execution and retention

The campaign freezes development seeds `0, 1, 2, 3, 4`, rejects more than two
million total steps, and preflights dataset, schedule, parameter, and persistent
state byte ceilings before an arm runs. Its validator recomputes the supplied
dataset digest and every seed's initial-parameter and schedule digests, binds
installed source plus live Python/JAX/NumPy/backend identity, validates all
per-arm resource receipts, requires the exact ordered roster, and recomputes
aggregates. Publication validation reruns every shard under the current source
and compares all deterministic receipt fields; wall-clock timing remains
telemetry only. Hashes bind consistency; they are not authenticated execution
attestation.

All rows remain `inconclusive` because this campaign has no registered decision
rule, and the create-only writer refuses replacement. Lower-level receipts retain
all outcomes, including negative outcomes. Run the full campaign only after the
measured source is settled, to a new append-only development path. Before a
scientific claim, the adapted mechanisms still need development screening,
full-horizon confirmation against the live incumbent, paper-protocol
qualification where feasible, and a separately frozen fresh-seed evaluation.
