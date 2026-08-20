# Matched Forager rerun preflight

`asi-forager-rerun-preflight` is a read-only, non-executing readiness check for
[issue #1584](https://github.com/elizaOS/asi/issues/1584).
It does not pull, build, load, or run an OCI image; prepare an output directory; read reward arrays;
or authorize a campaign. A blocked report exits 2 and is not an execution receipt.

The frozen comparison remains the stationary `ForagaxTwoBiomeLarge-v1` FOV9 task at 499,712
transitions. It pins the Forager paper at arXiv `2605.01131v1`, the official comparator source at
`steventango/continual-foragax-agents@9710f60fa30da5badc451ad7ce3ff296d5070830`,
`continual-foragax==0.55.0`, and Docker image ID
`sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768`.
The open schedule has 21 inferential candidates, tuning seeds 2,300,001–2,300,010, and 210 cells.
It is development selection only and permanently nonpromoting. Failures and negative outcomes are
append-only campaign outcomes.

## Image recovery audit

The frozen value is a Docker image **config digest** because the executors require Docker inspect's
`.Id` to equal it. It is not a registry manifest digest and cannot by itself form a resolvable
`repository@sha256:...` reference. The preflight audits three declared immutable identity records:
the official CPU qualification receipt and qualification record, and the RNG-parity receipt. They
contain the required config digest but no registry repository/manifest reference. The issue's
18 August 2026 UTC audit also found only receipts in GitHub code search and no published blob.

The issue records a non-substitutable reconstruction failure. Recovered source, lock, uv 0.9.24,
and Debian packages reproduced 120 symlinks, 6,564,533,833 regular-file bytes, and a
6,579,148,800-byte cache tar, but not the frozen identities. The reconstructed cache was
`bfe31dbbfef08b7ff14e6c4801a8ce50073bc8bf87991c5ef38541e905faf9be` with inventory
`b9c100b3b573f9c5dcf266c42a99bee9e2055b41b3552b398178619dff9337e3`; the required values are
`1a95d1e8e4a11b3be1db0deffed2ee354c76f7f1f47211a8bdbc7c3dccfa9b55` and
`ec400b9191dd336af5a66413b5ec0e9621b760c438de58e13a26d7e54e6dee1e`. Randomized uv
archive-v0 directory identities explain the mismatch. Rebuilding or editing the attestation would
create a different runtime and is forbidden by this frozen protocol.

Run the offline check from the repository root:

```bash
.venv/bin/asi-forager-rerun-preflight --project-root "$PWD"
```

The only OCI command it may issue is a local `docker image inspect` of the exact config digest.
It never issues `pull`, `load`, `build`, or `run`. On a host without Docker it records
`runtime_unavailable`; with Docker but no exact local image it records `image_absent`.

## Frozen accounting contract

The preflight recomputes 210 resets, 104,939,520 environment transitions and agent-action queries,
and 104,939,730 delivered observations (one reset observation plus one observation per transition
per cell). It requires future current-source qualification/cell receipts to bind environment resets
and transitions, observation delivery, action and model queries, optimizer updates, replay inserts
and samples, initial/final/peak persistent numeric bytes, raw result bytes, and elapsed nanoseconds.
Timing is telemetry-only.

Existing `ResourceAccounting` fields are useful disclosures but explicitly are not total memory or
compute. They cannot satisfy this new dynamic receipt contract without a reviewed schema addition
and a fresh qualification. Do not infer missing bytes or queries from parameter counts.

## Remaining blockers

The current report is expected to fail closed on all of these:

- the exact Docker image config digest is not in the local OCI store;
- no pinned registry `repository@manifest-digest` resolves that config digest;
- no fresh current-source v2 qualification manifest exists;
- current qualification does not emit the complete dynamic byte/query receipt contract; and
- no new append-only, explicitly nonpromoting output namespace has been declared.

After the exact original image or an authorized registry resolver is recovered, rerun qualification
from current source. A deliberately reviewed new runtime protocol is a different path; it must use
new identities and cannot claim equivalence to the missing image. Neither path may reuse or modify
the historical `2c3b214c` qualification/campaign roots or any sealed output.
