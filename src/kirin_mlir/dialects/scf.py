"""What the import checks on Kirin ``scf`` beyond the generic path.

Kirin's lowering ends the body of every ``scf.for`` and both bodies of every
``scf.if`` with a terminator, ``scf.yield`` or ``func.return``. A body
without one would be imported silently: ``scf`` is not registered, so MLIR
accepts any last operation an unregistered dialect may own, and Kirin's
verifier does not look. Kirin's interpreter then runs the block again without
end. The paper's fault injection found it: 67 deleted terminators passed
every check that does not need the original program, 52 of them hung the
interpreter, and no semantic oracle noticed 3.
"""

from __future__ import annotations

from kirin.dialects import scf

from ..import_ import require_terminators, register_statement_import

register_statement_import(scf.For)(require_terminators)
register_statement_import(scf.IfElse)(require_terminators)
