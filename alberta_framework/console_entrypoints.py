"""Lazy console-script boundaries for source-attested Forager commands.

Some Forager execution modules deliberately reject hard-linked implementation
sources. Package installers such as uv may legitimately hard-link wheel files
from a local cache, so importing those modules while loading console-script
metadata would reject an otherwise ordinary installation. These wrappers keep
entry-point discovery inert. Invoking a command still imports its original
implementation, where every staging and execution integrity check remains in
force.
"""

from __future__ import annotations


def forager_benchmark_main() -> int:
    """Load and run the Forager benchmark CLI."""

    from alberta_framework.forager_cli import main

    return main()


def historical_forager_main() -> int:
    """Load and run the historical Forager inspection CLI."""

    from alberta_framework.forager_cli import historical_main

    return historical_main()


def official_foragax_oci_main() -> int:
    """Load and run the official Foragax OCI CLI."""

    import sys

    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        from alberta_framework.benchmarks._official_foragax_oci_cli import (
            build_parser,
        )

        build_parser(
            base_image_default="",
            source_commit_default="",
            uv_binary_sha256_default="",
        ).parse_args()
        return 0

    from alberta_framework.benchmarks.official_foragax_oci import main

    return main()


__all__ = [
    "forager_benchmark_main",
    "historical_forager_main",
    "official_foragax_oci_main",
]
