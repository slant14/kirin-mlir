"""Kirin kernels made ready for MLIR: typed, free of closures, gathered into a module.

Issue 03. MLIR needs three things Kirin does not guarantee (``docs/assessment.md``,
gaps 2 and 4):

**A concrete type on every value.** ``@basic`` defaults to ``typeinfer=False``,
which leaves ``py.binop.add`` results as ``~T``. ``TypeInfer`` is run on every
kernel here, whatever its decorator said.

**No closures.** A closure reads its captured values through
``func.getfield(%self, i)``, and ADR 0001 drops ``%self`` at the boundary. That
is only sound if nothing still reads it.

**A module, not a method.** A Kirin pass applies to one ``Method``
(``kirin/passes/abc.py``), but a kernel calls other kernels through
``func.invoke``, whose callee is a ``Method`` object held in an attribute. MLIR
resolves calls by symbol within one module, so every kernel reachable from the
entry point has to be in it.

:func:`prepare_for_mlir` returns a :class:`PreparedModule`: copies of the entry
kernel and of every kernel it reaches, each typed and with its closure promoted.
It works on copies (``Method.similar``), so exporting never changes the kernel
the caller goes on running. The emitter accepts only a ``PreparedModule``; that
is how "refuse to emit an unprepared method" is enforced, by construction, the
same way :class:`kirin_mlir.names.NameTable` guarantees its names.

What Kirin's ``LambdaLifting`` actually does, checked against 0.22.16 on
2026-10-06, and narrower than ``docs/assessment.md`` gap 4 assumed: it promotes
a closure *object* whose captured values are compile-time constants into a
plain ``func.func``, replacing each ``func.getfield`` with a ``py.constant`` of
the captured value. Kirin's own passes already split such nested functions out
into separate kernels called through ``func.invoke``, so that is where they are
found and promoted here. A ``func.lambda`` *statement* left in a body is a
closure over values known only at run time. No rewrite in Kirin removes it,
and one that is returned or passed on cannot be removed by lambda lifting at
all, so it is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from kirin import ir
from kirin.passes import TypeInfer
from kirin.dialects import py, func
from kirin.dialects.func.rewrite.lambdalifting import LambdaLifting

from .errors import MLIRExportError


@dataclass(frozen=True)
class PreparedModule:
    """Kernels ready to emit as one MLIR module.

    Build it with :func:`prepare_for_mlir`, never directly.

    Args:
        methods: the prepared copies, entry kernel first, then every kernel it
            reaches, in the order they were found.
        sources: the kernels they were copied from, in the same order. Kept so
            that a ``func.invoke`` in a copy, which still names the original
            callee, can be resolved to a symbol.
    """

    methods: tuple[ir.Method, ...]
    sources: tuple[ir.Method, ...]

    @property
    def entry(self) -> ir.Method:
        """The prepared copy of the kernel that was exported."""
        return self.methods[0]

    @property
    def dialects(self) -> frozenset[ir.Dialect]:
        """Every dialect any kernel in the module uses.

        Kernels may be compiled with different dialect groups, so this is the
        set to build a :class:`kirin_mlir.names.NameTable` from.
        """
        return frozenset(d for m in self.methods for d in m.dialects.data)

    def symbol(self, method: ir.Method) -> str:
        """The MLIR symbol for a kernel, given its source or its prepared copy."""
        for prepared, source in zip(self.methods, self.sources):
            if method is prepared or method is source:
                return prepared.sym_name
        raise MLIRExportError(
            f"kernel {method.sym_name!r} is not in this module; it was not "
            "reachable from the entry kernel when the module was prepared"
        )


def prepare_for_mlir(method: ir.Method) -> PreparedModule:
    """Gather, copy, promote and type every kernel ``method`` reaches.

    Raises:
        MLIRExportError: if a kernel holds a closure over run-time values, or
            two kernels in the module share a symbol name.
    """
    sources: list[ir.Method] = []
    methods: list[ir.Method] = []
    pending = [method]
    while pending:
        source = pending.pop(0)
        if any(source is seen for seen in sources):
            continue  # reached twice, or recursion
        prepared = _prepare_one(source)
        sources.append(source)
        methods.append(prepared)
        pending.extend(_reachable(prepared))

    _check_symbols(methods)
    return PreparedModule(tuple(methods), tuple(sources))


def _prepare_one(source: ir.Method) -> ir.Method:
    prepared = source.similar()
    if isinstance(prepared.code, func.Lambda):
        # A nested function Kirin has already split out into its own kernel:
        # a closure object whose captures are constants in ``fields``. Kirin's
        # LambdaLifting has exactly this promotion, but only applies it to a
        # closure held by a py.constant, never to a func.invoke callee, so it
        # is called directly -- on the copy, never on the shared original.
        LambdaLifting()._promote_lambda(prepared)
    _refuse_closures(prepared)
    TypeInfer(dialects=prepared.dialects)(prepared)
    return prepared


def _refuse_closures(method: ir.Method) -> None:
    for stmt in method.callable_region.walk():
        if isinstance(stmt, func.Lambda):
            raise MLIRExportError(
                f"func.lambda {stmt.sym_name!r} in kernel {method.sym_name!r} is "
                "a closure over values known only at run time. Kirin's lambda "
                "lifting promotes closures whose captures are compile-time "
                "constants, and one that is returned or passed on cannot be "
                "lifted at all, so this kernel cannot be exported yet",
                stmt=stmt,
            )
        if isinstance(stmt, func.GetField):
            raise MLIRExportError(
                f"kernel {method.sym_name!r} still reads a captured value through "
                "%self after lambda lifting, and %self does not cross into MLIR "
                "(ADR 0001, decision 2)",
                stmt=stmt,
            )


def _reachable(method: ir.Method) -> Iterator[ir.Method]:
    """Kernels ``method`` refers to: callees, and kernels held as constants."""
    for stmt in method.callable_region.walk():
        if isinstance(stmt, func.Invoke):
            yield stmt.callee
        elif isinstance(stmt, py.Constant):
            payload = stmt.value
            if isinstance(payload, ir.PyAttr) and isinstance(payload.data, ir.Method):
                yield payload.data


def _check_symbols(methods: list[ir.Method]) -> None:
    owners: dict[str, ir.Method] = {}
    for method in methods:
        name = method.sym_name
        if not name:
            raise MLIRExportError(
                "a kernel in the module has no symbol name, so MLIR could not "
                "call it"
            )
        if name in owners:
            raise MLIRExportError(
                f"two different kernels in the module are both named {name!r}; "
                "MLIR resolves calls by symbol within a module, so they cannot "
                "both be exported. Rename one of them"
            )
        owners[name] = method
