"""Per-dialect ``emit.mlir`` method tables, for what generic form cannot say.

Importing this package registers them. Kept apart from :mod:`kirin_mlir.emit`
so the generic emitter imports no particular dialect, which is what lets it
emit dialects it has never seen.
"""

from . import func as func
