"""What the import checks on Kirin ``cf`` beyond the generic path.

A branch passes values to the block it names, one per block argument. MLIR
checks that for a branch it knows (``BranchOpInterface``), never for an
unregistered one, and Kirin's ``cf.br`` and ``cf.cond_br`` check nothing
(their ``verify`` is empty). So a mismatch would be imported silently and
fail only when the branch is taken.

It is not hypothetical. MLIR 23's ``--canonicalize`` erases the unused
arguments of a block whose predecessor is an unregistered terminator, and
leaves the terminator passing them: region simplification finds terminators
with ``hasTrait<IsTerminator>``, which an unregistered operation never has,
so it never marks those arguments live (``scripts/probe_mlir_irdl.py``,
section 12). The import now refuses the result instead of rebuilding it.
"""

from __future__ import annotations

from kirin import ir
from kirin.dialects import cf

from ..errors import MLIRImportError
from ..import_ import register_statement_import


def _check(node: ir.Statement, values, block: ir.Block, which: str) -> None:
    if len(values) != len(block.args):
        raise MLIRImportError(
            f"cf.{node.name} passes {len(values)} value(s) to its {which}, "
            f"which takes {len(block.args)}; a pass that cannot see through "
            "the unregistered branch, such as --canonicalize's region "
            "simplification, may have changed one and not the other"
        )


@register_statement_import(cf.Branch)
def check_branch(node: cf.Branch) -> None:
    _check(node, node.arguments, node.successor, "successor")


@register_statement_import(cf.ConditionalBranch)
def check_conditional_branch(node: cf.ConditionalBranch) -> None:
    _check(node, node.then_arguments, node.then_successor, "then successor")
    _check(node, node.else_arguments, node.else_successor, "else successor")
