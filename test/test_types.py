"""The Kirin lattice to ``!kirin.*`` mapping."""

import os
import sys
import subprocess

import pytest
from kirin import types
from kirin.ir.attrs.types import Union, Vararg, Generic, Literal, PyClass, TypeVar

from kirin_mlir import MLIRExportError, emit_type


def test_any_and_bottom():
    assert emit_type(types.Any) == "!kirin.any"
    assert emit_type(types.Bottom) == "!kirin.bottom"


def test_pyclass_uses_the_fully_qualified_name():
    assert emit_type(types.Int) == '!kirin.pyclass<"builtins.int">'
    assert emit_type(types.NoneType) == '!kirin.pyclass<"builtins.NoneType">'


def test_pyclass_ignores_kirins_display_sugar():
    """``!py.IList`` is how Kirin prints it; the identity is the real class."""
    ilist = pytest.importorskip("kirin.dialects.ilist")
    body = ilist.IListType.body
    assert body.display_name == "IList" and body.prefix == "py"
    assert emit_type(body) == ('!kirin.pyclass<"kirin.dialects.ilist.runtime.IList">')


def test_literal_keeps_both_value_and_declared_type():
    assert emit_type(Literal(3)) == (
        '!kirin.literal<3 : i64, !kirin.pyclass<"builtins.int">>'
    )
    # The datatype is passed explicitly: ``LiteralMeta`` caches on
    # ``(data, datatype)``, and in Python ``hash(True) == hash(1)`` with
    # ``True == 1``, so a bare ``Literal(True)`` returns a cached ``Literal(1)``
    # if any test constructed one first. That is upstream behaviour, not ours,
    # but a test that depends on it is a test that fails by execution order.
    assert emit_type(Literal(True, types.Bool)) == (
        '!kirin.literal<true, !kirin.pyclass<"builtins.bool">>'
    )
    # the declared type need not be the Python type of the datum
    assert emit_type(Literal(1, types.Float)) == (
        '!kirin.literal<1 : i64, !kirin.pyclass<"builtins.float">>'
    )


def test_union_members_are_sorted():
    one = emit_type(Union([types.Int, types.String]))
    other = emit_type(Union([types.String, types.Int]))
    assert one == other
    assert one == (
        '!kirin.union<[!kirin.pyclass<"builtins.int">, '
        '!kirin.pyclass<"builtins.str">]>'
    )


def test_union_order_does_not_depend_on_the_hash_seed():
    """``Union.types`` is a frozenset; emitted text must not vary run to run."""
    script = (
        "from kirin import types\n"
        "from kirin.ir.attrs.types import Union\n"
        "from kirin_mlir import emit_type\n"
        "print(emit_type(Union([types.Int, types.String, types.Float,"
        " types.Bool, types.NoneType])))\n"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, outputs


def test_kirin_collapses_degenerate_unions_before_we_see_them():
    assert emit_type(Union([])) == "!kirin.bottom"
    assert emit_type(Union([types.Int])) == '!kirin.pyclass<"builtins.int">'


def test_unresolved_typevar_becomes_any():
    assert emit_type(TypeVar("T")) == "!kirin.any"


def test_bounded_typevar_keeps_its_bound():
    assert emit_type(TypeVar("T", types.Int)) == '!kirin.pyclass<"builtins.int">'


def test_generic_carries_its_arguments_in_one_array():
    assert emit_type(types.List) == (
        '!kirin.generic<!kirin.pyclass<"builtins.list">, [!kirin.any]>'
    )
    assert emit_type(Generic(dict, types.String, types.Int)) == (
        '!kirin.generic<!kirin.pyclass<"builtins.dict">, '
        '[!kirin.pyclass<"builtins.str">, !kirin.pyclass<"builtins.int">]>'
    )


def test_generic_vararg_is_folded_into_the_argument_array():
    assert emit_type(types.Tuple) == (
        '!kirin.generic<!kirin.pyclass<"builtins.tuple">, '
        "[!kirin.vararg<!kirin.any>]>"
    )


def test_vararg_on_its_own():
    assert emit_type(Vararg(types.Int)) == (
        '!kirin.vararg<!kirin.pyclass<"builtins.int">>'
    )


def test_method_types():
    assert emit_type(types.MethodType) == "!kirin.anymethod"
    signature = types.MethodType[[types.Int, types.String], types.Bool]
    assert emit_type(signature) == (
        '!kirin.method<[!kirin.pyclass<"builtins.int">, '
        '!kirin.pyclass<"builtins.str">], '
        '!kirin.pyclass<"builtins.bool">>'
    )


def test_method_type_without_a_declared_result_is_unknown_not_none():
    from kirin.ir.attrs.types import FunctionType

    assert emit_type(FunctionType((types.Int,))) == (
        '!kirin.method<[!kirin.pyclass<"builtins.int">], !kirin.any>'
    )


def test_nesting_composes():
    nested = Generic(list, Union([types.Int, types.String]))
    assert emit_type(nested) == (
        '!kirin.generic<!kirin.pyclass<"builtins.list">, '
        '[!kirin.union<[!kirin.pyclass<"builtins.int">, '
        '!kirin.pyclass<"builtins.str">]>]>'
    )


def test_an_unknown_lattice_element_is_refused():
    class Foreign:
        pass

    with pytest.raises(MLIRExportError, match="no MLIR type mapping"):
        emit_type(Foreign())  # type: ignore[arg-type]


def test_a_class_with_no_module_identity_is_refused():
    anonymous = PyClass(type("Ghost", (), {}))
    anonymous.typ.__module__ = ""
    with pytest.raises(MLIRExportError, match="no identity outside this process"):
        emit_type(anonymous)
