# Low-cost activation and feature comparison (#1566)

This is a permanently nonpromoting development lane. It does not contain a
benchmark result or establish external state of the art. The executable is
`asi-activation-feature-ipmnist`; it runs one arm through the current IPMNIST
screening runner and only creates a new, immutable receipt path.

## Audited sources

- **Smooth-Leaky:** ICLR 2026 `XZf6wObHX4` and arXiv `2509.22562v4`
  (2026-04-30). The camera-ready paper discloses the official repository
  `lute47lillo/activations_plasticity`; the campaign pins commit
  `bdce354782cd183d63550819550b33312506d3e3`. The repository has no license
  file, so it is a read-only disambiguation source and no code is copied.
  Result-v1 has a fixed source-field contract; campaign plans separately bind
  this exact source registry. Equation 1 is
  `alpha*x + (1-alpha)*x*sigmoid(c*x/p)`. The development arm uses the paper's
  displayed `alpha=0.1, p=3, c=5` values. A fixed Leaky-ReLU arm removes the
  smooth transition while retaining the negative leak.
- **AID:** arXiv `2502.01342v2` (2025-06-15), ICML 2025. No official repository
  is disclosed by the paper, so Algorithm 2 is the pinned implementation
  source. The arm uses its simplified element-wise Bernoulli interval choice
  at `p=0.9`. The deterministic expectation and ordinary-dropout arms isolate
  stochastic interval assignment and negative-interval activation. Both
  stochastic arms draw the same number of Threefry Bernoulli variates.
- **Deep Fourier Features:** arXiv `2410.20634v1` (2024-10-27), ICLR 2025. No
  public official code revision is disclosed. The implementation follows the
  paper map `[sin(z), cos(z)]`. Half-width affine preactivations keep the next
  layer's activation width fixed. First-layer-only and sine-only arms test the
  depth and complementary-pair claims.

## Comparison contract and remaining gates

All arms inherit the same seed-derived permutations, example indices, EMA
input conditioner, SGD/decay update count, and pre/post-update metric queries
from `run_screening_config`. Each family has an explicit mechanism-off arm
that is bit-exact with the live `sgd_ema_norm_d099` control. This is the matched
causal control; the stronger `rls_head_resid_l1_preset005` incumbent remains a
separate live context comparator because its RLS head, resource shape, and
body-training signal are not a mechanism-off match.

The development seed set is frozen at `0, 1, 2, 3, 4`. A complete comparison
contains every registered arm exactly once for one seed; its validator requires
identical configuration, observation/update/query counts, parameter allocation,
persistent numeric bytes, and learner-visible information. The learner receives
the current example label but no task-boundary identifier. Receipts allow only
`supported`, `rejected`, or `inconclusive`, retain every outcome permanently,
and can never authorize scientific promotion.

The receipt keeps ASI whole-stream metrics separate and explicitly says that
no paper metric was reported. The papers use different optimizers, horizons,
batching, architectures, task schedules, and/or metrics. Deep Fourier Features
also changes the number of active affine parameters; both allocated and active
counts are recorded. Before any scientific comparison, development runs still
need paired multi-seed screening, full-horizon confirmation, resource-acceptable
comparison to the live incumbent, and a separately frozen fresh-seed protocol.

## Frozen campaign workflow

`asi-activation-feature-campaign` provides two separate, immutable plans. The
cheap screen is exactly 2 tasks × 500 examples; the full-horizon remeasurement
is exactly 200 tasks × 5,000 examples. Both plans require all 11 arms at all 5
development seeds (55 fresh-process shards each). The full plan does not become
smaller after seeing the cheap screen. The cheap plan uses the unconsumed
result-v1 seeds 0–4. The full plan preregisters the disjoint seeds
156610–156614 and uses result v2. Result-v1 validation enforces its fixed
seed and source-field contract.
These are still development seeds; neither stage can promote a claim.

Full-horizon execution is conditionally authorized. It requires a retained,
strictly valid cheap-screen aggregate from the exact same dataset, source, and
runtime, and at least one primary candidate (`smooth_leaky`, `aid`, or
`deep_fourier`) must have a simultaneous interval wholly above zero. If that
gate fails, the cheap negative/inconclusive result is retained and the full run
does not execute. If it passes, every full-stage arm and seed remains required;
the gate never authorizes a selected-candidate subset.

Each plan binds the exact MNIST bytes, current implementation sources,
Python/JAX/dependency/runtime identity, configuration, schedule-derived receipt
identity, resource schedule, and output namespace. Shard receipts use the
`asi.activation_feature_ipmnist.result.v1` contract for the cheap screen and
`asi.activation_feature_ipmnist.result.v2` for full confirmation. Every shard
binds an immutable plan digest, and aggregation does not reinterpret its
self-reported outcome. Run each `run-shard` command in a fresh Python process;
`summarize` rejects any roster other than the complete 55 unique shard files.

The eight predeclared candidate-versus-family-off comparisons use paired seed
deltas in whole-stream mean online accuracy. A two-sided Student-t interval
with 4 degrees of freedom uses Bonferroni alpha `0.05 / 8` (critical value
`5.261057575065803`). A simultaneous interval wholly above zero is
`supported`, wholly below zero is `rejected`, and every other result is
`inconclusive`. These are permanently nonpromoting development outcomes. The
aggregate retains every shard, decision, resource count, and negative outcome;
timing is telemetry only and consistency hashes are not execution attestation.

Canonical append-only namespaces are:

- `outputs/activation_feature_ipmnist/cheap_screen.v1/`
- `outputs/activation_feature_ipmnist/full_confirmation.v1/`

Create `plan.json`, write shards under `shards/`, then publish
`aggregate.json`. The CLI refuses to replace an existing path. No campaign has
been run or result produced by this implementation change.

For either stage, first create the canonical directories and plan:

```bash
stage=cheap_screen  # or full_confirmation
root="outputs/activation_feature_ipmnist/${stage}.v1"
mkdir -p "$root/shards"
.venv/bin/asi-activation-feature-campaign plan --stage "$stage"
```

Run each matrix cell as its own process. The following shell is illustrative;
production scheduling may parallelize the same commands without changing a
shard:

```bash
seeds="0 1 2 3 4"
extra=()
if [ "$stage" = full_confirmation ]; then
  seeds="156610 156611 156612 156613 156614"
  extra=(--cheap-aggregate \
    outputs/activation_feature_ipmnist/cheap_screen.v1/aggregate.json)
fi
for seed in $seeds; do
  for arm in smooth_leaky smooth_leaky_off smooth_leaky_fixed_leak \
    aid aid_off aid_expected ordinary_dropout deep_fourier deep_fourier_off \
    deep_fourier_first_layer deep_fourier_sine_only; do
    .venv/bin/asi-activation-feature-campaign run-shard \
      --stage "$stage" --seed "$seed" --arm "$arm" "${extra[@]}"
  done
done
.venv/bin/asi-activation-feature-campaign summarize --stage "$stage" \
  "${extra[@]}" "$root"/shards/*.json
.venv/bin/asi-activation-feature-campaign validate "$root/aggregate.json"
```
