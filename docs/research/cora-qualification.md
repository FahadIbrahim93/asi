# CORA continual-RL qualification

This lane pins Powers et al., *CORA: Benchmarks, Baselines, and Metrics as a Platform for
Continual Reinforcement Learning Agents*, CoLLAs 2022, arXiv `2110.10067v2`, and official code
`AGI-Labs/continual_rl@f2754bb282757829765beb4703f24b87efa13ff9` (MIT; audited 17 August 2026).
The code requires Python >=3.7, PyTorch >=1.7 and torchvision. Its environment families add ALE
and Atari ROMs, Procgen, MiniHack plus NLE, or AI2-THOR plus the `crl_alfred` fork and roughly 1 GB
of CHORES trajectories. None is installed or fetched by ASI's qualification CLI.

The pinned metric configuration uses six Atari games for two 50-million-step cycles (600M training
steps); six Procgen games for five 5-million-step cycles (150M), with 200 training levels and the
full level distribution for evaluation; fifteen MiniHack train/eval pairs for two 10-million-step
cycles (300M); and several three-task CHORES sequences at roughly one million steps per task.
CORA continuously evaluates every task, normalizes each task by its largest absolute observed
return, and reports isolated forgetting and zero-shot forward-transfer changes. Its orchestration
knows task boundaries and assigns action-space/task identifiers; whether a particular policy consumes
that identity must be audited per baseline.

ASI's executable `asi.cora_development.v1` slice is a deterministic two-action recurring bandit,
not an environment reproduction. Three reward tasks repeat for two cycles. The runner knows boundaries
to schedule training and continual evaluation, but candidate arms receive no task ID. Replay Q-learning
is paired with an exact-update-budget mechanism-off control that repeats the current transition; a
task-ID Q table is a privileged strong control excluded from candidates, and deterministic uniform
random is retained. Four consumed development seeds bind tie-breaking, task order, updates and replay
selection. Evaluation happens before training and after every block across every task.

Receipts exactly count training and evaluation environment steps, model queries, agent updates,
replay inserts/samples/peak bytes, persistent numeric bytes, logical calls and telemetry-only elapsed
nanoseconds. Validators recompute all counters and all three metric summaries. Results are permanently
nonpromoting, task-information use is explicit, and negative outcomes must be retained.

The separate `asi.cora.procgen.fixed_action_smoke.v1` contract is the first genuine external
qualification surface. It pins the official repository and commit through issue #1581's external
qualification authority and freezes the six-game Procgen sequence (Climber, Dodgeball, Ninja,
Starpilot, Bigfish, Fruitbot), 64x64 RGB observations, 15 actions, 200 training levels,
full-distribution evaluation, five cycles, 5 million training steps per task, and the paper's
250,000-step/20-window/20-seed metric parameters. These are catalog values, not a launch request.

An isolated provider must declare an immutable image, exact Python/PyTorch/torchvision/Gym/Procgen/
NumPy versions, a lock hash, a clean official checkout at the pinned commit, its Git tree, source
archive and install-tree hashes, hashes for required experiment/metrics entry points, and exact
Procgen distribution, compiled-data, install-tree, and license identities. The provider interface
emits one bounded training/evaluation reset-step pair for every game, using frozen training level
seeds 0-5 and disjoint evaluation level seeds 10000-10005. The host copies and hashes
exact int32 task/level/action arrays, uint8 observations, float32 rewards, and boolean termination
signals; enforces the task/split/fixed-action schedule; and records exact trace-array bytes.
External-runtime persistent numeric bytes are provider-reported, consistency-bound, and explicitly
unattested until a reviewed runtime-state inventory can derive them. The fixed action has no model
queries or learner updates, and task and boundary
identity are evaluator-only. The receipt is current-source/runtime bound and nonpromoting.

`asi-cora-external-qualification --blockers` only emits fail-closed metadata. It does not inspect
the network, download source, build an image, accept assets, launch Procgen, or write a result.
Content hashes are consistency bindings, not authenticated execution attestation. Until a supplied
official checkout, runtime, asset manifest, raw trace, and authorization are independently reviewed,
every external gate stays blocked and the native bandit remains explicitly not CORA.

Protocol gaps before external comparison include acquisition and independent verification of the
pinned source/runtime/Procgen identities; observation preprocessing, frame stacking, action
unification, episode truncation and stochastic seeding; actor/learner concurrency; continual-test
rollout counts and whether they affect learner state; task-ID and boundary exposure per baseline;
CLEAR replay bytes and replay ratio; EWC Fisher computation; P&C capacity; task-specific return
normalization; independent run aggregation; official checkpoint/result parity; full published step
budgets; matched ASI controls; hardware timing; and untouched preregistered scientific seeds. The
native slice's unnormalized binary metrics only test equations and information flow. No CORA result,
performance claim, or SOTA claim exists.
