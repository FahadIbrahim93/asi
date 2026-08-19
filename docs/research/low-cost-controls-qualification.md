# Low-cost activation and feature bounded comparator

This is an executable, permanently nonpromoting development comparator for
issue #1566. It compares Smooth-Leaky, interval-wise dropout (AID), and deep
Fourier features against a shared ReLU control on the bounded hidden-network
input-permuted-MNIST lane introduced by #1583. It is not a reproduction,
performance result, scientific evaluation, or SOTA claim.

## Primary-source audit

| mechanism | authoritative version | identity | preprint |
|---|---|---|---|
| Smooth-Leaky | ICLR 2026 | `ICLR-2026:XZf6wObHX4` | `arXiv:2509.22562v4` |
| Interval-wise dropout (AID) | ICML 2025, PMLR v267 pp. 47991–48026 | `ICML-2025:park25b` | `arXiv:2502.01342v2` |
| Deep Fourier features | ICLR 2025 | `ICLR-2025:NIkfix2eDQ` | `arXiv:2410.20634v1` |

- *Activation Function Design Sustains Plasticity in Continual Learning* — Lute
  Lillo, Nick Cheney. The camera-ready page header reads "Published as a
  conference paper at ICLR 2026"; the arXiv record is the preprint.
- *Activation by Interval-wise Dropout* — Sangyeon Park, Isaac Han, Seungwon Oh,
  Kyungjoong Kim. Final version is the ICML 2025 proceedings entry.
- *Plastic Learning with Deep Fourier Features* — Alex Lewandowski, Dale
  Schuurmans, Marlos C. Machado. The ICLR 2025 record is already the pinned
  version in `sota-landscape.md`.

## Official code

| mechanism | official code | status |
|---|---|---|
| Smooth-Leaky | [`lute47lillo/activations_plasticity`](https://github.com/lute47lillo/activations_plasticity) | **no license file present** |
| AID | none located | recorded absent |
| Deep Fourier features | none located | recorded absent |

The Smooth-Leaky repository is disclosed **only** in the camera-ready PDF's
embedded link annotations — not on the arXiv abstract page, the proceedings
listing, or a GitHub search by method name. It carries no license file, so it
grants no redistribution or derivative rights by default. The catalog records it
as located, unlicensed, and **not copied**; `LowCostCatalogEntry.validate()`
fails closed if any of those three facts is edited away.

For AID and deep Fourier features no author-published implementation was
located. Following the Normalize-and-Project precedent, the catalog records
official code as absent rather than pinning a third-party commit as if it were
authoritative.

## Mechanism definitions as pinned

**Smooth-Leaky** (Eq. 1), a `C^1` drop-in for Leaky ReLU keeping the negative
floor and positive-side identity while removing the kink:

    f(x) = a*x + (1 - a) * x * sigmoid(c * x / p)

with the paper's illustrated defaults `a = 0.1`, `p = 3.0`, `c = 5.0`.

**AID** (Algorithm 1) partitions the preactivation range into `k` disjoint
intervals covering `R`, each with dropout probability `p_j`; training draws
`m ~ Bernoulli(1 - p_j)` and testing scales by `(1 - p_j)`. The simplified
`AID_p` is ReLU at `p = 1` and a linear network at `p = 0.5`. Property 1 gives
the equivalent used here: ReLU with probability `p`, `min(x, 0)` otherwise.

**Deep Fourier features** replace each unit with a two-element sinusoid basis on
the same preactivation, in every layer:

    Fourier(z) = [sin(z), cos(z)]

Proposition 1 justifies it: for any `z`, one branch is within
`c = sqrt(2)*pi^2/28 ~= 0.05` of a linear function on `[z - pi/4, z + pi/4]`.

## Arms

Seven arms share one frozen seed, schedule, update rule, and observation count,
and differ only in the nonlinearity:

- `sgd_current_control` — ReLU;
- `smooth_leaky` / `smooth_leaky_off` (`alpha=1`);
- `aid` / `aid_off` (`relu_probability=1`);
- `deep_fourier` / `deep_fourier_off` (`feature_enabled=False`).

`aid_off` is ReLU by construction, so it must trace `sgd_current_control`
exactly. That equality is asserted in the lane tests and is what pins the
harness itself as mechanism-neutral: if it ever drifts, the harness moved the
numbers rather than the mechanism, and no other comparison in the lane is
trustworthy.

## Paper protocol differences and closed gates

Recorded in `LowCostCatalogEntry.protocol_differences` and enforced by
`validate()`:

- cumulative input-permuted MNIST, not the papers' class-incremental
  CIFAR/TinyImageNet, ALE, or non-stationary MuJoCo protocols;
- two width-bounded MLP hidden layers, not the papers' CNN/ResNet or Rainbow;
- per-example SGD, not the papers' minibatch Adam or RL optimizers;
- the deep Fourier arm halves its preactivation count so post-activation width
  equals the control and downstream weights keep the control's shape;
- AID uses the simplified two-interval form at one fixed probability, not a
  tuned interval schedule;
- Randomized Smooth-Leaky is out of scope; only the fixed-leak variant runs.

Deep Fourier features are the only mechanism here that changes layer geometry.
Holding post-activation width equal is the paper's own "reducing the effective
width of the layer"; the resulting parameter delta is reported through the arm's
`persistent_bytes` rather than presented as byte-matched.

Results from this lane are development-only and permanently nonpromoting, and
must not be presented as reproductions or as a numerical ranking against the
papers' reported curves.
