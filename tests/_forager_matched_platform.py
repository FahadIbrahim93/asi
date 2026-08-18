"""Platform predicates for the Linux-only matched-current publication primitives.

Matched-current publication deliberately depends on kernel primitives that only
Linux provides, and every call site fails closed elsewhere:

* ``open(O_TMPFILE)`` + ``linkat(AT_EMPTY_PATH)`` fill an inode that has no
  directory entry until it is complete.  ``forager_matched_campaign._publish_bytes``
  refuses with "O_TMPFILE is required for crash-safe publication".
* ``renameat2(RENAME_NOREPLACE)`` publishes a directory without ever replacing an
  existing destination.  ``forager_matched_seal._publish_verified_no_replace``
  refuses with "renameat2 is required for seal publication",
  ``forager_matched_campaign._rename_no_replace`` with "renameat2 is required for
  exclusive publication", and
  ``forager_matched_qualification._publish_directory_no_replace`` with "atomic
  no-replace publication is unavailable on this platform".
* ``/proc/self/fd`` names open descriptors, which the descriptor-leak censuses and
  the fsync-order tests read back.  Unlike the other two this can also be missing
  on Linux, so it is probed independently rather than inferred from the platform.

Each predicate runs the same probe as the guard it stands in for, so a gated test
skips exactly when the library would refuse and runs wherever the library
succeeds.  ``test_forager_matched_container_hardening`` already uses the
equivalent runtime skip for its procfs descriptor census.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest


def _probe_renameat2() -> bool:
    """Repeat the publication guards' own ``dlsym`` lookup for ``renameat2``."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:  # dlopen(NULL) is unavailable off POSIX; treat it as absent
        return False
    return getattr(libc, "renameat2", None) is not None


HAS_O_TMPFILE = bool(getattr(os, "O_TMPFILE", 0))
HAS_RENAMEAT2 = _probe_renameat2()
HAS_PROCFS_DESCRIPTORS = Path("/proc/self/fd").is_dir()

requires_o_tmpfile = pytest.mark.skipif(
    not HAS_O_TMPFILE,
    reason="crash-safe artifact publication requires Linux O_TMPFILE",
)
requires_renameat2 = pytest.mark.skipif(
    not HAS_RENAMEAT2,
    reason="atomic no-replace publication requires Linux renameat2",
)
requires_procfs_descriptors = pytest.mark.skipif(
    not HAS_PROCFS_DESCRIPTORS,
    reason="descriptor-path readback requires procfs",
)
