# COOM continual-Doom qualification

COOM is an external, isolated continual-RL benchmark. The native command
`asi-coom-qualification-smoke` does **not** run Doom: it exercises a bounded
synthetic pixel-shaped adapter contract and the current SARSA consumer while
leaving every external qualification gate closed. It writes no artifacts and
can never create benchmark or scientific evidence.

## Audited authority

- Paper: *COOM: A Game Benchmark for Continual Reinforcement Learning*, final
  NeurIPS 2023 Datasets and Benchmarks proceedings record
  `d61d9f4fe4357296cb658795fd7999f0`. There is no arXiv revision series to
  substitute for this final proceedings version.
- Official project: `https://github.com/TTomilin/COOM.git`, ASI pin
  `7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e`, MIT license. The current
  GitHub object was independently resolved as the 2026-03-05 commit “More
  robust plotting”; its setup and configuration files were inspected at that
  exact object. Source/asset hashing and execution remain explicit gates.
- Environment dependencies published by COOM 1.0.1: `vizdoom`,
  `opencv-python`, `scipy==1.11.4`, and `gymnasium==0.28.1`. Learning extras add
  `tensorflow==2.11`, `tensorflow-probability==0.19`, and `wandb`. These do not
  belong in ASI's JAX environment.

COOM has eight embodied, egocentric pixel scenarios. Pitfall and Arms Dealer
use 1,000-step episode caps; the other published scenarios use 2,500. Scenario
success signals differ—distance covered, weapons delivered, frames alive, or
kill count—so raw rewards are not interchangeable.

The two core eight-task orders are frozen in the machine catalog:

- CD8: Run-and-Gun Obstacles, Green, Resized, Monsters, Default, Red, Blue,
  Shadows.
- CO8: Pitfall, Arms Dealer, Hide and Seek, Floor is Lava, Chainsaw, Raise the
  Roof, Run and Gun, Health Gathering.

CD4/CO4 are the second halves of the corresponding eight-task sequences. At the
pin, CD16 repeats the CD8 environment variations and CO16 repeats the CO8
scenario order.
Paper experiments used seeds 0–9. The published learning defaults include
200,000 steps per environment, 50,000 replay entries, update warm-up 5,000,
batch size 128, frame skip 4, explicit task boundaries, and multi-head/task-ID
interfaces unless `--hide_task_id` changes the contract. The pinned wrapper
resizes to normalized 84×84 observations with four-frame stacks; evaluation
defaults to three episodes. Replay and optimizer reset at task changes by
default, while the critic does not. ASI must preserve
task-ID-visible and task-ID-hidden results as different protocols.

The primary COOM metrics are average performance, forgetting, and forward
transfer. Forward transfer requires matched from-scratch SAC runs on every task;
the native smoke therefore computes none of these metrics.

## Native contract smoke

The frozen smoke runs CD8 or CO8 with three consumed development seeds, four
bounded synthetic steps per task, an 8×8×3 `uint8` observation fixture, an
explicit task ID, and four arms:

1. deterministic cyclic adapter control;
2. ASI's live SARSA learner over pixels plus the allowed task ID;
3. mechanism off; and
4. fixed-action parity.

Mechanism-off and fixed-action receipts must have identical action, reward, and
observation hashes. The schema records reset/step/policy query counts and exact
observation, action, reward, terminal, task-ID, agent-state, and environment
state bytes. It also records dependency discoverability without importing any
external module. Synthetic rewards are retained only to verify causal adapter
plumbing and are never summarized as performance.

Use `asi-coom-qualification-smoke --catalog-only` to print the setup catalog or
run the default bounded contract smoke. No option launches COOM.

The separate `external_runtimes/coom/` directory now supplies a digest-pinned,
hash-locked Linux qualification image. Its fixed-action smoke executes the real
ViZDoom engine for two steps in each official CO8 task, verifies the pinned
source/license/WAD/config identity, and reports an unattested, nonpromoting
runtime receipt. A local audit ran the image twice at seed 1582000 and observed
the same trace SHA-256
`c74968494ccebaaeac4bc1e0c0f1db7546ac5091b831c05a4c0c727266da696f`.
Those receipts were not retained, so the digest is a reproducibility target,
not execution evidence; timing remains telemetry-only. The runtime closes the
bounded engine-load/reset/step implementation prerequisite, not the full
benchmark or learner qualification.

## Gates still open

- qualify any separately downloaded Doom data beyond the 33 WAD/config assets
  already bound by the source archive;
- lock the separate TensorFlow baseline environment, including CUDA and driver;
- verify exact CD/CO/COC/MIXED sequence enums, action spaces, observation
  preprocessing, reward normalization, termination/truncation, frame skip,
  seeds, evaluation cadence, task-ID visibility, and metric formulas at the pin;
- expand the bounded deterministic real-engine trace beyond the fixed-action
  CO8 qualification slice;
- freeze matched environment steps, gradient updates, replay bytes, evaluation
  queries, network bytes, accelerator memory, and timing;
- reproduce from-scratch SAC controls required for forward transfer and retain
  all negative outcomes; and
- only then consider an agent adapter, development comparison, or fresh-seed
  scientific protocol. Nothing here supports a SOTA or robotics claim.
