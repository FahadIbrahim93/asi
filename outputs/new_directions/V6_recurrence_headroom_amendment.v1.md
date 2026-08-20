# V6 append-only amendment and audit

This amendment preserves the original V6 JSON, runner, and report byte for
byte. The original runner checked family separation only for seed 0 before its
36 learner cells, although the registered control applied across the run.

The missing controls can be reconstructed without learner or benchmark
execution. Replaying only the deterministic permutation-schedule construction
shows 100 distinct M1 permutations and exactly five M4 permutations for each
of seeds 0, 1, and 2. Schedule digests are retained in the companion JSON.

The original JSON contains no Bayes result. The repository already retains a
200,000-sample Bayes summary for the same default Gaussian configuration and
exact seeds. Its per-seed accuracies are 0.983350, 0.988170, and 0.981415 with
binomial Monte Carlo SEMs 0.00028612, 0.00024176, and 0.00030199. The matched
mean is 0.9843116667, not the seed-0-only 0.9833 used in the original report.
Against the independently recomputed best M4 mean of 0.738268, the descriptive
difference is 0.2460436667.

The registered all-arm paired table remains primary. Every arm met the
registered criterion on all three consumed development seeds. That bounded
statement does not generalize beyond those seeds and does not show an explicit
recurrence-indexing mechanism. The original 7.9x grouping was post hoc, omitted
`upgd_raw`, and grouped distinct mechanisms; it is not retained as a causal or
primary result.

This micro-suite finding does not measure IPMNIST headroom: the IPMNIST
schedule uses a fresh permutation per task and contains no recurring mapping to
index. Complete historical runtime, lock, and exact argv identities were not
recorded, and the all-seed controls are reconstructed rather than having run
before the historical cells. The amended status is therefore explicitly
inconclusive, permanently nonpromoting development history.
