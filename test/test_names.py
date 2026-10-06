"""Kirin names mangled into names MLIR accepts (ADR 0002 decision 1)."""

import pytest
from kirin import ir
from kirin.decl import statement
from kirin.prelude import basic
from kirin.dialects import cf, scf, func
from kirin_dialects import public_dialects
from kirin.dialects.py import binop, constant

from kirin_mlir import PREFIX, NameTable, MLIRExportError, mangle


def _dialect(name: str, *stmt_names: str) -> ir.Dialect:
    """A throwaway dialect holding empty statements with the given names."""
    dialect = ir.Dialect(name)
    for stmt_name in stmt_names:
        statement(dialect=dialect)(
            type(f"Stmt_{len(dialect.stmts)}", (ir.Statement,), {"name": stmt_name})
        )
    return dialect


def test_mangle_replaces_dots_and_lowercases():
    assert mangle("py.binop") == "py_binop"
    assert mangle("const.none") == "const_none"
    assert mangle("SQRT_X") == "sqrt_x"
    assert mangle("Depolarize1") == "depolarize1"


def test_kirin_statements_get_prefixed_names():
    table = NameTable.of(basic)
    assert table.operation(binop.Add) == "kirin_py_binop.add"
    assert table.operation(constant.Constant) == "kirin_py_constant.constant"
    assert table.operation(func.Return) == "kirin_func.return"
    assert table.operation(func.ConstantNone) == "kirin_func.const_none"


def test_lookup_accepts_an_instance_as_well_as_a_class():
    @basic(typeinfer=True)
    def add_one(x: int) -> int:
        return x + 1

    table = NameTable.of(add_one.dialects)
    names = [table.operation(stmt) for stmt in add_one.callable_region.walk()]
    assert names == [
        "kirin_py_constant.constant",
        "kirin_py_binop.add",
        "kirin_func.return",
    ]


def test_names_shared_with_upstream_mlir_are_prefixed():
    """``func``, ``cf`` and ``scf`` clash with upstream MLIR, and silently."""
    table = NameTable.of([func.dialect, cf.dialect, scf.dialect])
    assert table.namespace(func.dialect) == "kirin_func"
    assert table.namespace(cf.dialect) == "kirin_cf"
    assert table.namespace(scf.dialect) == "kirin_scf"


def test_stim_style_names_are_lowercased():
    gates = _dialect("stim.gate", "CX", "SQRT_X")
    table = NameTable.of([gates])
    assert sorted(table.operations.values()) == [
        "kirin_stim_gate.cx",
        "kirin_stim_gate.sqrt_x",
    ]


def test_every_kirin_dialect_gets_legal_unique_names():
    """The guard that keeps this honest as Kirin moves.

    If a statement is added upstream whose name is illegal after mangling, or
    collides with a sibling, this fails naming it. Distinct statements must
    get distinct names, or the emitted text could not be read back.
    """
    dialects = public_dialects()
    table = NameTable.of(dialects)
    statements = [stmt for dialect in dialects for stmt in dialect.stmts]
    assert len(set(table.operations.values())) == len(statements)
    assert all(ns.startswith(PREFIX) for ns in table.namespaces.values())


@pytest.mark.parametrize(
    "stmt_name",
    ["bad-name", "_hidden", "two..dots", "é", ""],
    ids=repr,
)
def test_a_name_mlir_would_reject_is_refused(stmt_name):
    with pytest.raises(MLIRExportError, match="IRDL rejects"):
        NameTable.of([_dialect("test.illegal", stmt_name)])


def test_statements_that_mangle_alike_are_refused():
    """Lowercasing is not injective: ``CX`` and ``cx`` would become one op."""
    dialect = _dialect("test.case", "CX", "cx")
    with pytest.raises(MLIRExportError, match="both mangle to the MLIR operation"):
        NameTable.of([dialect])


def test_dialects_that_mangle_alike_are_refused():
    """Dots become underscores, so ``a.b`` and ``a_b`` would share a namespace."""
    with pytest.raises(MLIRExportError, match="both mangle to the MLIR namespace"):
        NameTable.of([_dialect("test.clash"), _dialect("test_clash")])


def test_a_statement_outside_the_table_is_refused():
    table = NameTable.of([binop.dialect])
    with pytest.raises(MLIRExportError, match="belongs to no dialect"):
        table.operation(func.Return)
    with pytest.raises(MLIRExportError, match="not in this name table"):
        table.namespace(func.dialect)
