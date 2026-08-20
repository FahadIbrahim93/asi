# COOM isolated qualification runtime

This directory builds a source- and dependency-pinned Linux runtime for one
bounded, fixed-action COOM/ViZDoom CO8 smoke. It does not install COOM into the
ASI environment and does not run a learner, compute benchmark metrics, reproduce
the paper, or authorize promotion.

The runtime uses a digest-pinned Python base, fetches the official COOM commit,
verifies the source archive, keeps the upstream MIT license, verifies all 33
WAD/config assets at execution, and installs only a hash-locked environment
subset. The one-line patch replaces an
undeclared legacy `gym.RewardWrapper` import with the declared
`gymnasium.RewardWrapper`; no environment logic or asset is changed.

`qualification-manifest.json` binds the base image, Dockerfile, dependency
lock, patch, and smoke-validator bytes. Before emitting a receipt, the smoke
reconstructs the upstream Git tree (reversing exactly that one import patch),
rehashes the license and all assets, enforces the locked package versions, and
strictly validates the ordered task/step trace, safe empty info subsets,
resources, claims, and independently repeated trace golden.

Build and execute from this directory:

```bash
docker build --tag asi-coom-qualification:development .
docker run --rm asi-coom-qualification:development > receipt.json
docker run --rm asi-coom-qualification:development > receipt.second.json
```

Validate a retained receipt against the same image-bound schema and input
manifest without starting COOM:

```bash
docker run --rm --volume "$PWD/receipt.json:/receipt.json:ro" \
  asi-coom-qualification:development --validate-receipt /receipt.json
```

Compare `trace_sha256`, not the telemetry-only elapsed time or platform string.
The smoke consumes seed 1582000, all eight official CO8 tasks, and two action-0
steps per task. A matching trace is a deterministic runtime qualification check,
not authenticated execution attestation or a COOM result. Local audit receipts
are not retained by this directory, and failures exit nonzero without a structured
negative-outcome receipt. Callers that need a durable qualification record must
publish the complete stdout receipt in a separately reviewed, append-only output
namespace.
