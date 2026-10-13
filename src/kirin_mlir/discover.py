"""Finding Kirin dialects by name, without importing anything dangerous.

The IRDL command line takes dialect names (``py.binop``), so it needs every
``ir.Dialect`` a package defines. Walking a package's modules is the only way
to find them, and one module must never be imported:
``kirin/dialects/math/_gen.py`` is a code generator that rewrites the installed
package's files when imported (``docs/assessment.md``, 2026-10-06 addendum). So
private (``_``-prefixed) modules are skipped. Every dialect is re-exported by
its public package, so nothing is lost.
"""

from __future__ import annotations

import pkgutil
import importlib
from collections.abc import Iterable

from kirin import ir

#: Where Kirin's own dialects live.
KIRIN_DIALECTS = "kirin.dialects"


def find_dialects(packages: Iterable[str] = (KIRIN_DIALECTS,)) -> dict[str, ir.Dialect]:
    """Every ``ir.Dialect`` reachable from the public modules of ``packages``.

    Keyed by Kirin dialect name. A module that needs an optional package
    which is not installed (Kirin's ``vmath`` needs numpy) is skipped, so the
    result is whatever this environment can import.
    """
    found: dict[str, ir.Dialect] = {}
    for name in packages:
        package = importlib.import_module(name)
        modules = [package]
        for info in pkgutil.walk_packages(
            getattr(package, "__path__", []), package.__name__ + "."
        ):
            if any(part.startswith("_") for part in info.name.split(".")):
                continue
            try:
                modules.append(importlib.import_module(info.name))
            except ImportError:
                continue
        for module in modules:
            for value in vars(module).values():
                if isinstance(value, ir.Dialect):
                    found[value.name] = value
    return dict(sorted(found.items()))
