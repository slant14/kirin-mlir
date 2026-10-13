"""Every dialect Kirin ships, for tests that must cover all of Kirin.

Used so that a dialect or statement added upstream is checked the day it
appears. The search itself is :func:`kirin_mlir.discover.find_dialects`, which
the IRDL command line uses too.
"""

from __future__ import annotations

from kirin import ir

from kirin_mlir.discover import find_dialects


def public_dialects() -> list[ir.Dialect]:
    """Every ``ir.Dialect`` reachable from Kirin's public dialect packages."""
    return list(find_dialects().values())
