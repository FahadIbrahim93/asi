# Low-cost activation and feature comparison (#1566)

This is a permanently nonpromoting development lane. It does not contain a
benchmark result or establish external state of the art. The executable is
`asi-activation-feature-ipmnist`; it runs one arm through the current IPMNIST
screening runner and only creates a new, immutable receipt path.

## Audited sources

- **Smooth-Leaky:** arXiv `2509.22562v4` (2026-04-30). The official repository
  points to the immutable anonymous snapshot
  `activations_plasticity-E431`; that snapshot identifier, rather than a
  mutable branch name, is the code revision bound by the lane. Equation 1 is
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
