"""The emitter retargeted at upstream MLIR, as an external oracle for its layout.

Issue 03's third acceptance criterion. Kirin's emitted dialects are
unregistered until issue 04, so ``mlir-opt`` only parses them: it checks that
the text is well formed, not that a branch passes its target the right number
of values or that a call names a function that exists. Upstream ``arith``,
``cf`` and ``func`` are registered and verified in full. So this subclass
keeps every layout decision of :class:`EmitMLIR` -- operand order, successor
placement, ``operandSegmentSizes``, block arguments, the dropped ``%self``,
symbol references -- and changes only what each part says, for the slice of
Kirin that has an exact upstream counterpart: integer arithmetic and
comparison, constants, ``func`` and ``cf``.

Test-only, and only ever an oracle (``CLAUDE.md``, core design decisions):
Kirin ``func`` and ``cf`` are not upstream ``func`` and ``cf`` in general, and
the package never emits these names. Anything outside the slice raises
``LookupError``, so a kernel cannot pass by being silently mistranslated.
"""

from __future__ import annotations

from collections.abc import Sequence

from kirin import ir, types
from kirin.dialects import cf, py, func
from kirin.ir.attrs.types import BottomType, FunctionType

from kirin_mlir import EmitMLIR

#: arith.cmpi's predicate enum: eq, ne, slt, sle, sgt, sge.
PREDICATES = {
    py.cmp.Eq: 0,
    py.cmp.NotEq: 1,
    py.cmp.Lt: 2,
    py.cmp.LtE: 3,
    py.cmp.Gt: 4,
    py.cmp.GtE: 5,
}

OPERATIONS = {
    py.binop.Add: "arith.addi",
    py.binop.Sub: "arith.subi",
    py.binop.Mult: "arith.muli",
    **{cmp: "arith.cmpi" for cmp in PREDICATES},
    func.Function: "func.func",
    func.Return: "func.return",
    func.Invoke: "func.call",
    func.Call: "func.call_indirect",
    cf.Branch: "cf.br",
    cf.ConditionalBranch: "cf.cond_br",
}


def _holds_method(node: ir.Statement) -> bool:
    value = node.attributes.get("value")
    return isinstance(value, ir.PyAttr) and isinstance(value.data, ir.Method)


class UpstreamOracle(EmitMLIR):
    """Kirin's arithmetic slice as upstream ``arith``, ``cf`` and ``func``."""

    def operation_name(self, node: ir.Statement) -> str:
        if isinstance(node, py.constant.Constant):
            return "func.constant" if _holds_method(node) else "arith.constant"
        name = OPERATIONS.get(type(node))
        if name is None:
            raise LookupError(f"{type(node).__name__} is outside the oracle's slice")
        return name

    def type_text(self, typ: ir.Attribute) -> str:
        if isinstance(typ, FunctionType) and typ.return_type is not None:
            params = ", ".join(self.type_text(t) for t in typ.params_type)
            return f"({params}) -> ({self.type_text(typ.return_type)})"
        if isinstance(typ, types.TypeAttribute) and not isinstance(typ, BottomType):
            # Bool first: Kirin's lattice has bool <= int, as Python does.
            if typ.is_subseteq(types.Bool):
                return "i1"
            if typ.is_subseteq(types.Int):
                return "i64"
        raise LookupError(f"type {typ} is outside the oracle's slice")

    def attribute_entries(self, node: ir.Statement) -> dict[str, str]:
        entries = super().attribute_entries(node)
        if isinstance(node, func.Function):
            signature = node.signature
            inputs = ", ".join(self.type_text(t) for t in signature.inputs)
            output = self.type_text(signature.output)
            return {
                "sym_name": entries["sym_name"],
                "function_type": f"({inputs}) -> ({output})",
            }
        if isinstance(node, func.Invoke):
            return {"callee": entries["callee"]}
        if isinstance(node, func.Call):
            if node.kwargs:
                raise LookupError("func.call_indirect has no keyword arguments")
            return {}
        if isinstance(node, py.constant.Constant):
            return {"value": entries["value"]}
        if type(node) in PREDICATES:
            return {"predicate": f"{PREDICATES[type(node)]} : i64"}
        return entries

    def result_values(self, node: ir.Statement) -> Sequence[ir.SSAValue]:
        # Kirin's func.func defines the method as a value; upstream's defines
        # only a symbol.
        if isinstance(node, func.Function):
            return ()
        return super().result_values(node)


class KeepsSelf(UpstreamOracle):
    """The oracle without Kirin ``func``'s emit table, so ``%self`` is kept.

    The negative control: it shows that the oracle notices when the entry
    block disagrees with the signature, which is what dropping ``%self``
    (ADR 0001, decision 2) prevents. A different key means none of the
    ``emit.mlir`` method tables apply.
    """

    keys = ("emit.mlir.keeps_self",)
