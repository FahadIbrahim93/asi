# FTL Online Agent external-reproduction gate

This runbook advances the external FTL Online Agent / Continual Bench comparison without treating
ASI's bounded analogue as a reproduction. It is a development-readiness procedure only. It does
not authorize downloads, benchmark execution, evidence promotion, or changes to `outputs/`.

## Reviewed primary-source pins

- Paper: Liu et al., *Continual Reinforcement Learning by Planning with Online World Models*,
  PMLR 267:38397–38423 and arXiv `2507.09177v1`.
- Official environment repository: `https://github.com/sail-sg/ContinualBench`.
- Reviewed repository revision: `a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad`.
- The repository exposes `continual_bench.envs.ContinualBenchEnv`, MuJoCo sources, and tracked
  environment assets. Its `pyproject.toml` directly declares only `glfw==2.5.0`.
- The paper reports six tasks in the order `pick-place`, `button-press`, `door-open`, `peg-unplug`,
  `window-close`, `faucet-close`; 600 episodes; and 10–15 hours per run using one A100 and 16 CPUs.

These facts do not form a complete protocol lock. In particular, the official repository does not
publish the paper's OA world-model/planner implementation or a transitive runtime lock. ASI must
not infer that source from the environment package or substitute its native implementation while
describing the result as an external reproduction.

## Offline source qualification

Obtain the repository through a separately approved process and check out the exact revision. Do
not point the gate at a branch name or a modified tree. Then run:

```bash
.venv/bin/asi-ftl-external-readiness --checkout /absolute/path/to/ContinualBench
```

The command is read-only. It does not import or execute the checkout. It verifies the exact HEAD,
requires a clean tree including no untracked files, rejects symlinks and submodules, checks the
reviewed package metadata and required environment paths, hashes every tracked file and the asset
subset with SHA-256, binds the Git commit/tree objects, and records the current Python executable
and relevant installed distribution versions. It prints one canonical JSON report and returns `2`
while any reproduction gate is open. It never writes a result artifact.

Qualification binds local bytes to the reviewed commit; it is not an authenticated attestation of
where the checkout was obtained. The receipt therefore records
`repository_origin_attested=false`. Git inspection disables checkout-configured filesystem-monitor
hooks and lazy fetching, and tracked files are opened component-by-component without following
symlinks. These restrictions are part of the no-import/no-execution/read-only boundary.

An `official_checkout_invalid:` blocker is a hard failure. Preserve the checkout and report for
audit, correct the external material out of tree, and rerun. Never weaken the qualifier to accept a
different revision, a dirty tree, missing assets, or looser dependency metadata.

## Gates before any execution

The following remain required even after the checkout qualifies:

1. Obtain an author-published OA/planner implementation or independently implement and review the
   paper equations and algorithms. Record which route is used; do not label an independent port as
   the official implementation.
2. Freeze a complete paper protocol manifest: sparse encoder/model constants, FTL update timing,
   CEM horizon/population/iterations/elites, colored-noise and memory settings, action bounds,
   resets, termination, task switches, evaluation cadence, success thresholds, AP and regret
   definitions, and all seeds.
3. Build an immutable transitive environment lock for MuJoCo, Gymnasium, Meta-World-derived code,
   MBRL `v0.2.0`, Torch, NumPy/SciPy, GLFW, drivers, and accelerator runtime. The one direct GLFW
   pin in the environment repository is insufficient.
4. Audit the provenance, checksums, and license/re-distribution status of every XML, mesh, texture,
   and other MuJoCo asset.
5. Implement the paper OA and matched Perfect Memory/deep continual-learning controls under one
   runner. Match planner calls and information access; keep the privileged simulator oracle out of
   candidate comparisons.
6. Qualify exact counters for environment transitions, model updates, model queries, CEM candidate
   rollouts, peak/persistent numeric bytes, CPU/GPU memory, and accelerator-hours. Timing remains
   telemetry until separately qualified.
7. Run a cheap development smoke on consumed development seeds, retain negative results, and audit
   replay. Only then preregister an explicitly nonpromoting full development run. Freeze untouched
   scientific seeds in a separate protocol only if a claim warrants it.

Until all seven gates close, `execution_ready=false`, `reproduction_claim_allowed=false`, and
`external_results_present=false` are load-bearing. The historical FTL evidence chain and the
ASI-native closed-loop development lane remain separate.
