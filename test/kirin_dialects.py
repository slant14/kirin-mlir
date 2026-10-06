"""Every dialect Kirin ships, found without importing anything dangerous.

Used by tests that must cover all of Kirin rather than one dialect group, so
that a dialect or statement added upstream is checked the day it appears.
"""

from __future__ import annotations

import pkgutil
import importlib

from kirin import ir


def public_dialects() -> list[ir.Dialect]:
    """Every ``ir.Dialect`` reachable from Kirin's public dialect packages.

    Private (``_``-prefixed) modules are never imported:
    ``kirin/dialects/math/_gen.py`` is a code generator that rewrites the
    installed package's files when imported. Each dialect is re-exported by its
    public package, so skipping them loses nothing.
    """
    import kirin.dialects

    found: dict[str, ir.Dialect] = {}
    for info in pkgutil.walk_packages(
        kirin.dialects.__path__, kirin.dialects.__name__ + "."
    ):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            # a dialect needing an optional package that is not installed
            continue
        for value in vars(module).values():
            if isinstance(value, ir.Dialect):
                found[value.name] = value
    return sorted(found.values(), key=lambda d: d.name)
