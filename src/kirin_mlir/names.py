"""Kirin dialect and statement names, mangled into names MLIR accepts.

ADR 0002 decision 1. A Kirin statement is named
``dialect.name + "." + name`` -- ``py.binop.add``, ``stim.gate.SQRT_X`` -- and
real MLIR refuses both. MLIR 23's IRDL verifier allows only lowercase letters,
digits and underscores, with no leading underscore and no two in a row (its
messages: "name of operation must contain only lowercase letters, digits and
underscores", "should not contain leading or double underscores"). So every
name is mangled::

    MLIR namespace  = "kirin_" + mangle(dialect.name)    py.binop  -> kirin_py_binop
    MLIR operation  = mangle(stmt.name)                   SQRT_X    -> sqrt_x
    mangle(s)       = s.replace(".", "_").lower()

The ``kirin_`` prefix goes on *every* dialect, not only the four whose names
are also upstream MLIR dialects (``cf``, ``func``, ``scf``, ``math``). MLIR
accepts an IRDL dialect that reuses an upstream name without any error, so the
rule has to make clashes impossible rather than detectable. A prefix applied
only to today's clashes would make the emitted text depend on which MLIR was
installed.

Mangling is not injective: ``a.b`` and ``a_b`` meet, and lowercasing loses
case. So names are handed out only by :class:`NameTable`, which refuses a set
of dialects whose names are illegal or collide. Holding a ``NameTable`` is the
proof that every name in it is safe to emit; the emitter (issue 03) and the
IRDL generator (issue 04) both build one rather than calling :func:`mangle`
themselves.

``scripts/probe_mlir_irdl.py`` reproduces the MLIR behaviour this rests on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable

from kirin import ir

from .types import DIALECT
from .errors import MLIRExportError

#: Every emitted namespace starts with this: the type dialect's own name plus
#: an underscore, so no mangled namespace can equal ``kirin`` itself.
PREFIX = f"{DIALECT}_"

#: What MLIR 23's IRDL verifier accepts for a name, checked against real
#: ``mlir-opt``. Dialect names may also not start with a digit, but every
#: namespace starts with ``PREFIX``, so one rule covers both.
_LEGAL = re.compile(r"(?!_)(?!.*__)[a-z0-9_]+")


def mangle(name: str) -> str:
    """Replace dots with underscores and lowercase.

    Does not check the result. A mangled name may still be illegal or clash
    with another; :class:`NameTable` is what checks.
    """
    return name.replace(".", "_").lower()


def _checked(name: str, what: str) -> str:
    mangled = mangle(name)
    if not _LEGAL.fullmatch(mangled):
        raise MLIRExportError(
            f"cannot name {what} in MLIR: {name!r} mangles to {mangled!r}, which "
            "IRDL rejects; names may contain only lowercase letters, digits and "
            "underscores, with no leading or doubled underscore"
        )
    return mangled


@dataclass(frozen=True)
class NameTable:
    """Checked MLIR names for one set of Kirin dialects.

    Build it with :meth:`of`, never directly: ``of`` is where the checks are.

    Args:
        namespaces: Kirin dialect name to MLIR namespace.
        operations: statement class to full MLIR operation name,
            ``"namespace.op"``.
    """

    namespaces: dict[str, str]
    operations: dict[type[ir.Statement], str]

    @classmethod
    def of(cls, dialects: ir.DialectGroup | Iterable[ir.Dialect]) -> NameTable:
        """Name every statement of ``dialects``, refusing anything unsafe.

        Raises:
            MLIRExportError: if a mangled name is not a legal IRDL name, if two
                statements of one dialect mangle to the same operation, or if
                two dialects mangle to the same namespace.
        """
        if isinstance(dialects, ir.DialectGroup):
            dialects = dialects.data

        namespaces: dict[str, str] = {}
        owners: dict[str, ir.Dialect] = {}
        operations: dict[type[ir.Statement], str] = {}

        # Sorted so that, of several problems, the same one is reported first
        # on every run.
        for dialect in sorted(set(dialects), key=lambda d: d.name):
            namespace = PREFIX + _checked(dialect.name, f"dialect {dialect.name!r}")
            other = owners.get(namespace)
            if other is not None:
                raise MLIRExportError(
                    f"Kirin dialects {other.name!r} and {dialect.name!r} both "
                    f"mangle to the MLIR namespace {namespace!r}"
                )
            owners[namespace] = dialect
            namespaces[dialect.name] = namespace

            claimed: dict[str, type[ir.Statement]] = {}
            for stmt in dialect.stmts:
                op = _checked(stmt.name, f"statement {dialect.name}.{stmt.name}")
                rival = claimed.get(op)
                if rival is not None:
                    raise MLIRExportError(
                        f"statements {dialect.name}.{rival.name} and "
                        f"{dialect.name}.{stmt.name} both mangle to the MLIR "
                        f"operation {namespace}.{op}"
                    )
                claimed[op] = stmt
                operations[stmt] = f"{namespace}.{op}"

        return cls(namespaces, operations)

    def namespace(self, dialect: ir.Dialect) -> str:
        """The MLIR namespace for ``dialect``."""
        namespace = self.namespaces.get(dialect.name)
        if namespace is None:
            raise MLIRExportError(
                f"dialect {dialect.name!r} is not in this name table; build the "
                "table from every dialect the method uses"
            )
        return namespace

    def operation(self, stmt: ir.Statement | type[ir.Statement]) -> str:
        """The full MLIR operation name for a statement or statement class.

        Looked up by exact class, like :mod:`kirin_mlir.values`: a subclass of
        a registered statement is not that statement.
        """
        cls = stmt if isinstance(stmt, type) else type(stmt)
        name = self.operations.get(cls)
        if name is None:
            raise MLIRExportError(
                f"statement class {cls.__name__} belongs to no dialect in this "
                "name table; build the table from every dialect the method uses",
                stmt=None if isinstance(stmt, type) else stmt,
            )
        return name
