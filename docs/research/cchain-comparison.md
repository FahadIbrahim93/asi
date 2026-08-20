# C-CHAIN matched development comparison (#1565)

This is a permanently nonpromoting development adaptation of C-CHAIN (ICML
2025, arXiv `2506.00592v1`) and the authors' official implementation at commit
`2f8bedfefb6a0a276d7709447a5cdb75cecfdaad`. It is not an official-RL
reproduction, an IPMNIST result, scientific evidence, or a state-of-the-art
claim.

## Registered comparison

`asi-cchain-matched-development` executes the exact frozen development roster:
seeds `0, 1, 2, 3, 4` crossed with the charged mechanism-off Adam control, full
adaptive C-CHAIN penalty, orthogonal-only churn gradient, and projective-only
churn gradient. Every arm receives the same Threefry-derived parameter
initialization, input permutations, and example schedule. The learner sees the
current example label and receives no task identifier or task-boundary signal.

The campaign binds the exact supplied dataset, current installed execution
sources, Python and dependency runtime, JAX configuration/devices, schedules,
and initial states. Per-shard receipts account for task and churn-reference
updates, task/churn/NTK model queries, persistent numeric state, and the bounded
full-logit NTK Jacobian envelope. Timing is telemetry only. The aggregate
reports per-arm mean metrics and summed deterministic resources.

The strict validator rejects incomplete or reordered rosters, changed source,
runtime, dataset, schedules, initial states, resources, metrics, and arbitrary
campaign decisions. It reruns every shard under the current source and compares
all deterministic receipt fields. Campaign outcomes remain `inconclusive`
because no decision rule is registered. The create-only writer replays before
using atomic no-replace publication, so every output, including a negative
measurement, must use a new path. Consistency hashes are not authenticated
execution attestation.

## Remaining gates

The adaptation uses online single-example IPMNIST, a 32-example prior-stream
ring, one-update snapshots, a static 50-update coefficient window, ASI's MLP
and matched Adam constants, and a full-logit empirical NTK. The paper evaluates
continual RL with PPO or DoubleDQN, independently shuffled batches, mutable
model history, different networks and optimizer settings, and policy outputs.

Run this campaign only after the measured source is settled. Development still
requires the actual paired run, retained per-seed review, resource-acceptable
comparison with the live incumbent, and negative-result recording. Any paper
claim additionally requires qualification on official RL tasks/code. Any ASI
scientific claim requires a separately frozen protocol, untouched fresh seeds,
a versioned evidence artifact, and explicit promotion review.
