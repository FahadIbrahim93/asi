# JEPA-WM / V-JEPA 2-AC transfer feasibility

This is the permanently nonpromoting native feasibility lane for issue `#1577`.
It tests one narrow architectural question: can an encoder learned from earlier
ASI transitions be frozen, transferred into a fresh action-conditioned
predictor, and consumed by a live controller under explicit costs?

The matched development roster uses five consumed seeds, `1577000..1577004`.

## Independently verified references

- V-JEPA 2, `arXiv:2506.09985v1`, submitted 2025-06-11; official
  `facebookresearch/vjepa2` commit
  `204698b45b3712590f06245fbfba32d3be539812`. The paper pretrains on more than
  one million hours of internet video, then post-trains V-JEPA 2-AC on under 62
  hours of DROID robot video. The pinned repository exposes 300M–1B parameter
  V-JEPA 2 checkpoints and a ViT-g/16 action-conditioned checkpoint.
- JEPA-WM, `arXiv:2512.24497v3`, revised 2026-05-18 and accepted at TMLR;
  official `facebookresearch/jepa-wms` commit
  `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`. It studies visual
  state-action training, representation-space planning, and simulated and
  real-world navigation/manipulation, reporting gains over DINO-WM and
  V-JEPA-2-AC.

Both exact GitHub commit pages and repository trees were inspected. The pins
are provenance only; external code, checkpoints, and datasets are not imported
or executed by this lane.

## Official visual-token qualification boundary

`asi-vjepa-external-qualification` reports the unresolved official V-JEPA 2-AC
gates without downloading, importing, or executing external assets. The
corresponding provider contract accepts bounded `uint8` video clips and
`float32` actions plus robot-state/proprioception tensors, binds the pinned MIT
source tree/license separately from checkpoint terms, web-video and robot-video
manifests, full pretraining costs, isolated
runtime, and provider executable, and requires separate context, predicted
target, and EMA-target visual-token tensors. Its 256-pixel, 16-by-16-patch
geometry matches the pinned official AC loader; three context frames provide
three action and three state tokens for prediction of the fourth frame. Exact
input/output bytes, semantic queries, checkpoint loads, persistent bytes, and a
conservative resident-tensor peak-memory lower bound are validated. Smoke inputs
use an explicit Threefry
root and every receipt binds the current ASI source/runtime identity.

This is an executable adapter boundary, not an in-tree V-JEPA implementation.
Only a separately authorized isolated provider can exercise it with official
code and assets. A passing adapter smoke remains nonpromoting and does not show
paper parity, planning quality, robot competence, or authenticated execution.
The frozen no-imported-pretraining ablation keeps the visual architecture and
adapter geometry while using only ASI-generated or separately authorized visual
data, zero imported checkpoint/data bytes, and matched online-query,
peak-memory, and evaluation workloads; it explicitly does not pretend the
pretraining compute is matched.

## Native transfer experiment

The runner stores a bounded ASI-only `SwitchingTwoStateMDP` transition trace,
takes one trainable-encoder pass over it, and transfers only the learned encoder
into a freshly initialized predictor. That predictor then selects actions from
predicted rewards during an A/B recurrence. Frozen seeds, environment horizon,
pretraining trace length, warm-up, and exploration schedule are matched.

The roster separates:

1. encoder-only ASI transfer;
2. no-pretraining with the same frozen random-feature architecture;
3. a permuted transferred-encoder causal control;
4. a full encoder-and-predictor warm-start ceiling;
5. transferred encoder with its decision interface disabled;
6. exact no-model mechanism-off; and
7. the current SARSA agent as a strong live control.

Decision-off must reproduce mechanism-off action and reward hashes exactly.
Every arm reports environment/pretraining steps, pretraining examples and
updates (including applied encoder updates), imported-pretraining bytes (always zero), stored pretraining replay
bytes, online replay bytes (always zero), semantic encoder/model/control query
counts, encoder and total persistent bytes, environment bytes, and scoped
`perf_counter_ns` telemetry for pretraining, decisions, online updates, and
environment/control execution. Timing is process-local telemetry only, never an acceptance metric.
Semantic query counts describe public model operations; they are not FLOP or
kernel-launch estimates. Hashes are consistency bindings, not execution proof.

Run `asi-jepa-transfer-feasibility`. It prints one strictly validated JSON
receipt and never writes `outputs/`. Negative outcomes remain recorded and the
schema forbids scientific promotion or visual/robotics parity claims.

The complete campaign uses `asi-jepa-transfer-matched --output NEW.json`. It
binds the entire current package Python-source tree; Python, JAX, dependency,
device, and relevant environment identity; the exact workload and paper pins;
explicit Threefry roots; every ASI-only pretraining replay; environment and
mechanism initial states; pretraining/deployment transition, action, phase, and
decision schedules; the five-seed-by-seven-arm roster; and complete semantic
resource receipts. It explicitly records absent imported checkpoints, visual
datasets, robot datasets, and imported-pretraining bytes. Process-clock fields
in learner state are charged as persistent bytes but canonicalized to zero only
for the algorithmic initial-state hash.

The strict validator is input-pure and reruns all 35 shards. It recomputes every
result and resource except process-local timing values, whose schema and scope
remain validated but whose magnitudes are telemetry-only. Publication is atomic
and create-only. The campaign decision is always exactly `inconclusive`, even if
one arm leads every seed. Adding this machinery does not create a campaign result.

## Explicit gaps

This two-state, one-hot, one-step JAX lane is not a visual JEPA reproduction. It
has no video masking/tokenization, transformer encoder, target-encoder EMA,
image-goal energy planner, multistep latent rollout, DROID/web-video data,
imported checkpoint, camera calibration, or physical robot. It also lacks
paper-exact datasets, preprocessing, model scales, planners, horizons, metrics,
and seeds. External feasibility still requires an isolated PyTorch/CUDA runtime,
asset checksums and licenses, exact checkpoint/data/pretraining receipts,
matched planning budgets, qualified accelerator memory/FLOPs/latency, longer
recurrence and stochastic-retention tests, untouched evaluation seeds, and
robot safety/veto, control-frequency, sim-to-real, and hardware gates.

These five development seeds can never be reused by a future scientific or
promotion protocol.
