"""Kirin attributes to MLIR attributes, and the shape of a refusal."""

import pytest
from kirin import ir, types
from kirin.prelude import basic
from kirin.dialects import func

from kirin_mlir import (
    MLIRExportError,
    emit_attribute,
    register_attribute,
    emit_statement_attributes,
)
from kirin_mlir.attrs import ATTRIBUTE_HANDLERS


@pytest.fixture
def kernel():
    @basic(typeinfer=True)
    def add_one(x: int) -> int:
        return x + 1

    return add_one


def test_pyattr_payloads_pass_straight_through():
    assert emit_attribute(ir.PyAttr("outer")) == '"outer"'
    assert emit_attribute(ir.PyAttr(0)) == "0 : i64"
    assert emit_attribute(ir.PyAttr(True)) == "true"
    assert emit_attribute(ir.PyAttr(("x", "y"))) == '["x", "y"]'


def test_pyattr_with_a_declared_type_the_payload_contradicts_is_refused():
    attr = ir.PyAttr(1, pytype=types.Float)
    with pytest.raises(MLIRExportError, match="declares type"):
        emit_attribute(attr)


def test_pyattr_declared_any_asserts_nothing_so_it_passes():
    assert emit_attribute(ir.PyAttr(1, pytype=types.Any)) == "1 : i64"


def test_a_type_in_attribute_position_is_the_type_text():
    assert emit_attribute(types.Int) == '!kirin.pyclass<"builtins.int">'


def test_dialect_attributes_are_refused_until_a_dialect_registers_them(kernel):
    """``Signature`` is registered by ``kirin_mlir.dialects.func`` (issue 03).

    With that registration taken away, the general mapper refuses it: it holds
    no special case for any dialect.
    """
    signature = kernel.code.attributes["signature"]
    assert isinstance(signature, func.Signature)
    handler = ATTRIBUTE_HANDLERS.pop(func.Signature)
    try:
        with pytest.raises(MLIRExportError, match="no MLIR attribute mapping"):
            emit_attribute(signature)
    finally:
        ATTRIBUTE_HANDLERS[func.Signature] = handler


def test_a_schema_declared_type_is_carried_by_the_statement(kernel):
    """``slots`` declares ``tuple[str, ...]``, exactly as its field does."""
    slots = kernel.code.attributes["slots"]
    assert isinstance(slots, ir.PyAttr) and slots.type != types.PyClass(tuple)
    with pytest.raises(MLIRExportError, match="declares type"):
        emit_attribute(slots)
    assert emit_statement_attributes(kernel.code)["slots"] == '["x"]'


def test_a_declared_type_the_schema_does_not_give_is_still_refused(kernel):
    constant = next(
        stmt for stmt in kernel.callable_region.walk() if stmt.name == "constant"
    )
    constant.attributes["value"] = ir.PyAttr(1, pytype=types.Float)
    with pytest.raises(MLIRExportError, match="declares type"):
        emit_statement_attributes(constant)


def test_a_refusal_names_the_statement_and_the_key(kernel):
    constant = next(
        stmt for stmt in kernel.callable_region.walk() if stmt.name == "constant"
    )
    constant.attributes["scratch"] = ir.PyAttr(range(3))

    with pytest.raises(MLIRExportError) as caught:
        emit_statement_attributes(constant)

    message = str(caught.value)
    assert "statement 'py.constant.constant'" in message
    assert "attribute 'scratch'" in message
    assert caught.value.stmt is constant
    assert caught.value.key == "scratch"


def test_statement_attributes_emit_as_a_dict(kernel):
    constant = next(
        stmt for stmt in kernel.callable_region.walk() if stmt.name == "constant"
    )
    assert emit_statement_attributes(constant) == {"value": "1 : i64"}


def test_register_attribute_is_the_extension_point(kernel):
    saved = dict(ATTRIBUTE_HANDLERS)
    try:

        @register_attribute(func.Signature)
        def _emit_signature(attr: func.Signature) -> str:
            return "#test.signature"

        assert emit_attribute(kernel.code.attributes["signature"]) == (
            "#test.signature"
        )
    finally:
        ATTRIBUTE_HANDLERS.clear()
        ATTRIBUTE_HANDLERS.update(saved)


def test_a_non_attribute_is_refused():
    with pytest.raises(MLIRExportError, match="no MLIR attribute mapping"):
        emit_attribute(object())  # type: ignore[arg-type]
