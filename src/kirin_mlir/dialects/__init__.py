"""Per-dialect ``emit.mlir`` method tables, for what generic form cannot say.

Importing this package registers them, and the import hooks of
:func:`kirin_mlir.register_statement_import`. Kept apart from
:mod:`kirin_mlir.emit` so the generic emitter imports no particular dialect,
which is what lets it emit dialects it has never seen.
"""

from . import cf as cf, scf as scf, func as func
