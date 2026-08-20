# Reference-life development scorecard runbook

This runbook governs the manual 144-shard GitHub Actions scorecard. It is an
operational checklist, not authorization to run it. The campaign is development-only
and permanently nonpromoting. A completed campaign cannot populate `reference-dev`,
support a scientific or SOTA claim, or reuse these consumed seeds for promotion.

## Authorization and dispatch

1. Obtain explicit authorization for one exact 40-character commit that is the current
   `elizaOS/asi` `main` head. Do not dispatch while source changes are still expected.
2. Confirm the authorized commit contains
   `.github/workflows/reference-life-scorecard-dev.yml` and that the workflow has not
   already been dispatched for the intended campaign.
3. Dispatch `reference-life development scorecard` with that exact commit as
   `launch_sha`. Record the GitHub run ID immediately.
4. Do not rerun jobs or rerun the workflow run. The workflow admits only
   `github.run_attempt == 1`; shard artifact names intentionally identify one run ID,
   not a mixture of attempts. After an infrastructure or workflow failure, retain the
   failed run URL for the audit trail and start a fresh dispatch with a new run ID. A
   fresh dispatch still uses consumed development seeds and remains nonpromoting.

The shard CLI uses exit status `0` for a completed record and `1` for a valid retained
failure record. The workflow validates and uploads both. Any other status, missing
record, malformed record, identity mismatch, or incomplete 144-record roster fails
closed. A valid aggregate may therefore report `valid_execution_failure`; that is a
diagnostic result, not a reason to discard the campaign record.

## Download and independent validation

GitHub Actions retention is only a 90-day transfer window. Download the complete
validated campaign promptly from the run's artifact named
`reference-life-scorecard-<launch-sha>-<run-id>-validated`. Do not treat the individual
shard artifacts as the durable record when the complete campaign artifact exists.

Validate from a clean checkout of the exact authorized commit with its locked Python
3.12.12 environment. Place the download under a temporary `scorecard/` directory so
the receipt paths retain their recorded names, then run:

```bash
.venv/bin/python -m alberta_framework.benchmarks.reference_life_scorecard \
  validate /absolute/staging/scorecard/artifact.json

for shard in /absolute/staging/scorecard/shards/*.json; do
  .venv/bin/python -m alberta_framework.benchmarks.reference_life_scorecard \
    validate "$shard"
done
```

Require exactly 144 shard files. Independently recompute every byte count and SHA-256
in `run-receipt.v1.json`, then recompute `campaign_inventory_sha256` from the receipt's
canonical compact JSON inventory. Confirm the receipt names:

- repository `elizaOS/asi`, dispatch ref `refs/heads/main`, the authorized source and
  workflow commit, and `github_run_attempt` `1`;
- the observed run ID, 144 jobs, both frozen environments, all six frozen arms, and
  seeds 70000 through 70011;
- the checked-out Git tree, workflow blob, `uv.lock`, Python 3.12.12, and uv 0.9.24.

The scorecard validator binds the complete artifact and every shard to one current
source, runtime, and dependency identity. Those identities are consistency hashes, not
authenticated execution attestation. Timing and hardware peak memory remain
telemetry-only.

## Append-only publication and review

After validation, propose the complete downloaded bytes at this new namespace:

```text
outputs/reference_life_scorecard/development/
  github-run-<run-id>-<40-character-launch-sha>/
    plan.json
    artifact.json
    run-receipt.v1.json
    shards/
      <all 144 canonical shard JSON files>
```

The destination must not already exist. Never overwrite, amend, delete, rename, or
partially populate a published run namespace. Publish all 147 files together in a
reviewed PR, preserving their bytes. The review must repeat the validation and receipt
inventory checks, link the GitHub run, state the aggregate status exactly, and confirm
that no overview document calls the result scientific evidence, external SOTA, or a
promotion. If review rejects the record, retain the run URL and rejection reason; do
not silently replace it with a rerun.
