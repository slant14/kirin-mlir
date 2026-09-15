"""The single error this package raises when Kirin IR cannot cross into MLIR.

The rule from ``docs/adr/0001-mlir-interop.md`` is that unrepresentable input
**raises** rather than being widened or boxed opaquely. Silent widening produces
MLIR that verifies while meaning something different from the Kirin program it
came from, which is a silent failure in the one place this project exists to be
trustworthy.

The mapping functions in :mod:`kirin_mlir.types` and :mod:`kirin_mlir.values`
see a bare lattice type or a bare Python payload; they do not know which
statement or which attribute key it came from. The caller does. So the error
carries optional ``stmt`` and ``key`` slots that an outer frame fills in on the
way out -- see :func:`kirin_mlir.attrs.emit_statement_attributes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kirin import ir


class MLIRExportError(Exception):
    """Raised when a Kirin construct has no faithful MLIR representation.

    Args:
        message: what could not be represented, and why.
        stmt: the Kirin statement the construct was found on, if known.
        key: the attribute key the construct was found under, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        stmt: ir.Statement | None = None,
        key: str | None = None,
    ) -> None:
        self.message = message
        self.stmt = stmt
        self.key = key
        super().__init__(self._render())

    def attach(
        self,
        *,
        stmt: ir.Statement | None = None,
        key: str | None = None,
    ) -> MLIRExportError:
        """Fill in whichever context slots are still empty, in place.

        Mutating rather than re-raising a fresh exception keeps the original
        traceback, which points at the exact mapping function that refused.
        Slots already set win: the innermost frame that knows the context is
        the one that is right about it.
        """
        if self.stmt is None:
            self.stmt = stmt
        if self.key is None:
            self.key = key
        self.args = (self._render(),)
        return self

    def _render(self) -> str:
        where: list[str] = []
        if self.stmt is not None:
            where.append(f"statement {_stmt_name(self.stmt)!r}")
        if self.key is not None:
            where.append(f"attribute {self.key!r}")
        if where:
            return f"{self.message} [in {', '.join(where)}]"
        return self.message


def _stmt_name(stmt: Any) -> str:
    """Best-effort mnemonic for a statement, tolerant of non-statement input."""
    dialect = getattr(stmt, "dialect", None)
    name = getattr(stmt, "name", None)
    if name is None:
        return type(stmt).__name__
    if dialect is not None and getattr(dialect, "name", None):
        return f"{dialect.name}.{name}"
    return str(name)
