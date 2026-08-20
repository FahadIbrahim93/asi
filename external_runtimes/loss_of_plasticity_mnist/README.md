# Loss-of-plasticity MNIST qualification runtime

This directory defines a prospective, Linux/amd64, CPU-only runtime for issue
#1583's first official-code qualification gate. It fetches the exact official
source archive, installs a hash-locked Python 3.8/Torch 2.1 dependency set, and
verifies source, license, configuration, complete installed-distribution,
runtime, and plan identities as a non-root user without build-step network
access.

It does not download MNIST, execute an experiment, emit a qualification
receipt, create an `outputs/` artifact, establish paper/code parity, retain a
negative outcome, or authorize any later run. The image entry point only
revalidates the prospective plan and runtime.

The official MNIST README invokes `python3.8`, while the same current source
revision pins SciPy 1.11.2, whose supported Python floor is 3.9. This runtime
uses SciPy 1.10.1, the final Python-3.8-compatible release line, and records the
change as a nonparity compatibility deviation. The current code revision also
postdates the paper and applies cumulative input permutations; labels remain
the original MNIST digits and must never be described as random labels.

Any future runtime invocation must use a digest-pinned built image with network
disabled, a read-only root filesystem and dataset mount, a bounded no-execute
tmpfs, approved CPU/memory/PID/wall-clock/output limits, an exact separately
approved MNIST archive, and a NEW output path. Those gates are intentionally
not implemented or authorized here.
