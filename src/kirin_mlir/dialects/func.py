"""What Kirin ``func`` needs beyond the generic emitter.

Registered onto Kirin's own ``func`` dialect from outside the Kirin repository,
the same way ``bloqade-circuit`` registers ``emit.stim`` (``CLAUDE.md``, "How
this attaches to Kirin"). Importing :mod:`kirin_mlir` imports this module, so
the registration is in place before anything is emitted.

``func.func`` drops ``%self``
    Every Kirin ``func.func`` entry block starts with a ``MethodType``
    argument, ``%self``, that has no MLIR meaning (ADR 0001, decision 2). It is
    left out of the emitted entry block, so the block's arguments line up with
    the ``signature`` attribute. That is only sound if nothing reads it, so a
    function that still reads ``%self`` is refused here, by name, rather than
    reaching ``mlir-opt`` as an undeclared value. The import puts it back
    (:func:`restore_self`), read off the signature.

``func.lambda`` is refused
    A closure reads its captures through its own ``%self``.
    :func:`kirin_mlir.prepare_for_mlir` promotes or refuses every closure, so
    one reaching the emitter means the module was not built by it.

``Signature`` is a function type
    The ``signature`` attribute is spelled as a builtin MLIR function type in
    attribute position, ``(!kirin.pyclass<"builtins.int">) -> (...)``: the
    obvious candidate named in issue 03, and one ``mlir-opt`` parses. It keeps
    Kirin's key, ``signature``, so the import direction (issue 06) needs no
    renaming table, and :func:`import_signature` reads it back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import ir, types, interp
from kirin.dialects import func

from .. import parse
from ..attrs import register_attribute, register_attribute_import
from ..types import emit_type, import_type
from ..errors import MLIRExportError, MLIRImportError
from ..import_ import register_statement_import

if TYPE_CHECKING:
    from ..emit import EmitMLIR, MLIRFrame


@register_attribute(func.Signature)
def emit_signature(attr: func.Signature) -> str:
    inputs = ", ".join(emit_type(typ) for typ in attr.inputs)
    return f"({inputs}) -> ({emit_type(attr.output)})"


@register_attribute_import(func.Signature)
def import_signature(attr: parse.Attribute) -> func.Signature:
    """The ``Signature`` :func:`emit_signature` wrote: its inverse."""
    if not (
        isinstance(attr, parse.TypeAttr)
        and isinstance(attr.type, parse.FunctionType)
        and len(attr.type.results) == 1
    ):
        raise MLIRImportError(
            "a signature is written as a function type with one result, "
            "(inputs) -> (output)"
        )
    return func.Signature(
        inputs=tuple(map(import_type, attr.type.inputs)),
        output=import_type(attr.type.results[0]),
    )


@register_statement_import(func.Function)
def restore_self(node: func.Function) -> None:
    """Put back the ``%self`` argument :meth:`FuncMLIREmit.function` dropped.

    Its type is the function's own, read off the signature as Kirin's
    ``Method.self_type`` reads it, and its name hint is ``<sym_name>_self``,
    as Kirin's lowering names it. Nothing in the body uses it: the export
    refused a function that did.
    """
    blocks = node.body.blocks
    if not blocks:
        raise MLIRImportError(
            f"func.func {node.sym_name!r} has no entry block to put %self back in"
        )
    signature = node.signature
    self_type = types.FunctionType(signature.inputs, signature.output)
    name = f"{node.sym_name}_self"
    valid = ir.SSAValue.name_pattern.fullmatch(name)
    blocks[0].args.insert_from(0, self_type, name if valid else None)


@func.dialect.register(key="emit.mlir")
class FuncMLIREmit(interp.MethodTable):

    @interp.impl(func.Function)
    def function(self, emit: EmitMLIR, frame: MLIRFrame, node: func.Function):
        blocks = node.body.blocks
        if not blocks or not blocks[0].args:
            raise MLIRExportError(
                f"func.func {node.sym_name!r} has no %self entry argument to "
                "drop, so its signature cannot be lined up with its arguments",
                stmt=node,
            )
        self_arg = blocks[0].args[0]
        if self_arg.uses:
            raise MLIRExportError(
                f"func.func {node.sym_name!r} still reads %self, which does not "
                "cross into MLIR (ADR 0001, decision 2)",
                stmt=node,
            )
        return emit.emit_operation(frame, node, omit=(self_arg,))

    @interp.impl(func.Lambda)
    def lambda_(self, emit: EmitMLIR, frame: MLIRFrame, node: func.Lambda):
        raise MLIRExportError(
            f"func.lambda {node.sym_name!r} reached the emitter; a closure "
            "cannot be exported, and prepare_for_mlir refuses one, so this "
            "module was not built by prepare_for_mlir",
            stmt=node,
        )
