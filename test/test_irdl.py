"""IRDL generated from Kirin's statement declarations, checked by real ``mlir-opt``.

Issue 04. The positive checks load the generated IRDL and verify issue 03's
emitted modules against it. The negative checks are what show the leaf
dialects are *registered* rather than tolerated. Each corrupted module is
rejected with ``--irdl-file`` and accepted without it, so it is the IRDL that
catches the fault.
"""

import os
import sys
import subprocess

import pytest
from kirin import ir
from oracle import MLIR_OPT, SKIP_REASON, mlir_opt
from test_emit import EVERY, branches
from kirin.decl import info, statement
from kirin.prelude import basic
from kirin.dialects import py

from kirin_mlir import (
    TYPE_SCHEMA,
    TypeSchema,
    MLIRExportError,
    emit_mlir,
    exclusion,
    find_dialects,
    generate_irdl,
    prepare_for_mlir,
)
from kirin_mlir.cli import main

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)


@basic(typeinfer=True)
def mixed(x: int) -> float:
    """``lhs``, ``rhs`` and ``result`` all declare ``T``, and differ."""
    return x + 1.5


@basic(typeinfer=True)
def pair(x: int, y: int):
    """``py.tuple.new``, whose operands its fields do not declare."""
    return (x, y)


CORPUS = [*EVERY, mixed, pair]


@basic(typeinfer=True)
def mixed_pair(x: int):
    """``py.tuple.new(int, float)``: one variadic group, two types."""
    return (x, 1.5)


def ids(kernel: ir.Method) -> str:
    return kernel.sym_name


def hybrid(text: str, irdl: str, *flags: str) -> subprocess.CompletedProcess:
    """ADR 0002's verification: leaf dialects registered, control flow not."""
    return mlir_opt(
        text,
        f"--irdl-file={irdl}",
        "--allow-unregistered-dialect",
        *flags,
    )


@pytest.fixture
def irdl_for(tmp_path):
    """Write the IRDL for a prepared module's dialects; return its path."""

    def write(module) -> str:
        path = tmp_path / "kirin.irdl.mlir"
        path.write_text(generate_irdl(module.dialects).text)
        return str(path)

    return write


# -- the command line ---------------------------------------------------------


@needs_mlir
def test_the_cli_emits_irdl_mlir_loads(tmp_path):
    """Issue 04's first criterion, and a check that the ops are really declared."""
    out = tmp_path / "slice.irdl.mlir"
    assert main(["irdl", "py.binop", "py.constant", "py.cmp", "-o", str(out)]) == 0

    assert mlir_opt("", f"--irdl-file={out}").returncode == 0
    one_operand = (
        '%0 = "kirin_py_constant.constant"() {value = 1 : i64} : () -> !kirin.any\n'
        '%1 = "kirin_py_binop.add"(%0) : (!kirin.any) -> !kirin.any\n'
    )
    result = mlir_opt(one_operand, f"--irdl-file={out}")
    assert result.returncode != 0
    assert "expects exactly 2 operands" in result.stderr


@needs_mlir
def test_an_excluded_dialect_is_reported_and_not_emitted(tmp_path, capsys):
    """Issue 04's second criterion: ``kirin-mlir irdl func``."""
    out = tmp_path / "func.irdl.mlir"
    assert main(["irdl", "func", "-o", str(out)]) == 1
    assert "func.func has a region" in capsys.readouterr().err

    # Not declared, so MLIR still sees kirin_func as an unregistered dialect.
    result = mlir_opt('"kirin_func.return"() : () -> ()\n', f"--irdl-file={out}")
    assert result.returncode != 0
    assert "unregistered dialect" in result.stderr


def test_an_unknown_dialect_is_a_usage_error(capsys):
    assert main(["irdl", "py.nosuch"]) == 2
    assert "py.nosuch" in capsys.readouterr().err


def test_all_takes_every_dialect_and_only_reports_exclusions(tmp_path, capsys):
    out = tmp_path / "all.irdl.mlir"
    assert main(["irdl", "--all", "-o", str(out)]) == 0
    assert "not registering func" in capsys.readouterr().err


def test_python_dash_m_runs_the_cli():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-m", "kirin_mlir", "irdl", "py.unary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "irdl.dialect @kirin_py_unary" in result.stdout


# -- verifying issue 03's output ----------------------------------------------


@needs_mlir
@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_every_emitted_module_verifies_against_its_irdl(kernel, irdl_for):
    """Issue 04's third criterion, positive half."""
    module = prepare_for_mlir(kernel)
    text = emit_mlir(module)
    result = hybrid(text, irdl_for(module), "--verify-roundtrip")
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


def _drop_an_operand(module):
    add = next(
        s for s in module.entry.callable_region.walk() if isinstance(s, py.binop.Add)
    )
    add.args = add.args[:1]
    return emit_mlir(module)


def _drop_an_attribute(module):
    constant = next(
        s
        for s in module.entry.callable_region.walk()
        if isinstance(s, py.constant.Constant)
    )
    del constant.attributes["value"]
    return emit_mlir(module)


def _malform_a_type(module):
    return emit_mlir(module).replace(
        '!kirin.pyclass<"builtins.int">', "!kirin.pyclass<42 : i64>"
    )


def _use_a_builtin_type(module):
    """Every ``int`` written as ``i64``: well formed, but not a Kirin type."""
    return emit_mlir(module).replace('!kirin.pyclass<"builtins.int">', "i64")


@needs_mlir
@pytest.mark.parametrize(
    "corrupt, error",
    [
        (_drop_an_operand, "expects exactly 2 operands"),
        (_drop_an_attribute, 'attribute "value" is expected'),
        (_malform_a_type, "expected base attribute 'builtin.string'"),
        (_use_a_builtin_type, "'i64' does not satisfy the constraint"),
    ],
    ids=[
        "wrong operand count",
        "missing attribute",
        "malformed !kirin type",
        "builtin type",
    ],
)
def test_a_corrupted_module_is_rejected_only_because_of_the_irdl(
    corrupt, error, irdl_for
):
    """Issue 04's third criterion, negative half, and issue 03's pending control.

    The same text passes with ``--allow-unregistered-dialect`` alone, so it is
    registration that rejects it, not a parse error.
    """
    module = prepare_for_mlir(branches)
    irdl = irdl_for(module)
    text = corrupt(module)

    assert mlir_opt(text, "--allow-unregistered-dialect").returncode == 0
    result = hybrid(text, irdl)
    assert result.returncode != 0
    assert error in result.stderr


@needs_mlir
def test_a_shared_typevar_is_not_a_shared_constraint(irdl_for):
    """Kirin types ``x + 1.5`` as ``add(int, float) -> float``.

    ``lhs``, ``rhs`` and ``result`` all declare ``T``. One shared IRDL variable
    would mean "equal" and reject this valid program.
    """
    module = prepare_for_mlir(mixed)
    add = next(
        s for s in module.entry.callable_region.walk() if isinstance(s, py.binop.Add)
    )
    assert add.lhs.type != add.rhs.type
    result = hybrid(emit_mlir(module), irdl_for(module))
    assert result.returncode == 0, result.stderr


@needs_mlir
def test_a_variadic_group_mixing_types_is_rejected_though_kirin_accepts_it(irdl_for):
    """A known false rejection, pinned so a newer MLIR that lifts it is noticed.

    IRDL binds a constraint to the first type it meets, and a variadic group
    has one constraint for all its operands, so its operands must share a
    type. Kirin's ``(x, 1.5)`` does not. The generated file's header says so,
    and a C++ dialect lifts it (issue 07 phase A): ODS's ``Variadic<AnyType>``
    does not bind.
    """
    module = prepare_for_mlir(mixed_pair)
    text = emit_mlir(module)
    assert mlir_opt(text, "--allow-unregistered-dialect").returncode == 0
    result = hybrid(text, irdl_for(module))
    assert result.returncode != 0
    assert "but got '!kirin.pyclass<\"builtins.float\">'" in result.stderr
    assert "Checked too strictly" in generate_irdl(module.dialects).text


# -- what the generator reports -----------------------------------------------


def test_exactly_the_control_flow_dialects_are_excluded():
    """ADR 0002 decision 2, over all of Kirin. A new exclusion needs a look."""
    excluded = generate_irdl(find_dialects().values()).excluded
    assert {e.dialect for e in excluded} == {"cf", "func", "module", "scf"}


def test_an_exclusion_names_the_statement_responsible():
    from kirin.dialects import cf, func

    assert str(exclusion(func.dialect)) == "func.func has a region"
    reason = exclusion(cf.dialect)
    assert reason is not None and reason.statement.startswith("cf.")
    assert exclusion(py.binop.dialect) is None


def test_the_report_lists_traits_exclusions_and_loose_shapes():
    """Issue 04's fourth criterion: a reader can see what was not expressed."""
    generated = generate_irdl(find_dialects().values())
    assert "Pure" in generated.traits["py.binop.add"]
    assert generated.loose["py.tuple.new"] == ("operands",)
    assert generated.loose["py.unpack.unpack"] == ("results",)

    report = [line for line in generated.text.splitlines() if line.startswith("//")]
    assert any("py.binop.add" in line and "Pure" in line for line in report)
    assert any("func.func has a region" in line for line in report)
    assert any("py.tuple.new" in line for line in report)


@needs_mlir
def test_all_of_kirin_loads(tmp_path):
    path = tmp_path / "all.irdl.mlir"
    path.write_text(generate_irdl(find_dialects().values()).text)
    result = mlir_opt("", f"--irdl-file={path}")
    assert result.returncode == 0, result.stderr


# -- generated, not hand-written ----------------------------------------------


def _shape_dialect(arguments: int) -> ir.Dialect:
    """A fresh dialect whose one statement takes ``arguments`` operands."""
    dialect = ir.Dialect("test_shape")
    namespace = {
        "__annotations__": {
            **{f"x{i}": ir.SSAValue for i in range(arguments)},
            "result": ir.ResultValue,
        },
        **{f"x{i}": info.argument() for i in range(arguments)},
        "result": info.result(),
        "name": "op",
    }
    statement(dialect=dialect)(type("Op", (ir.Statement,), namespace))
    return dialect


@needs_mlir
def test_adding_an_argument_changes_the_irdl_with_no_hand_edits(tmp_path):
    """Issue 04's fifth criterion. Only the statement changes; MLIR notices."""
    two_operands = (
        '%0 = "test.a"() : () -> !kirin.any\n'
        '%1 = "kirin_test_shape.op"(%0, %0) : (!kirin.any, !kirin.any) -> !kirin.any\n'
    )
    for arguments, accepted in ((1, False), (2, True)):
        path = tmp_path / f"shape{arguments}.irdl.mlir"
        path.write_text(generate_irdl([_shape_dialect(arguments)]).text)
        result = mlir_opt(
            two_operands, f"--irdl-file={path}", "--allow-unregistered-dialect"
        )
        assert (result.returncode == 0) == accepted, result.stderr


def test_an_unknown_type_parameter_kind_is_refused(monkeypatch):
    """The type schema and the generator must not drift apart silently."""
    monkeypatch.setitem(
        TYPE_SCHEMA, "pyclass", TypeSchema("pyclass", ("name: Mystery",), "")
    )
    with pytest.raises(MLIRExportError, match="Mystery"):
        generate_irdl([py.binop.dialect])


def test_a_hand_written_init_with_declared_operands_is_trusted():
    """``py.ilist.new`` builds exactly its fields, so it is not loosened."""
    generated = generate_irdl(find_dialects().values())
    assert "py.ilist.new" not in generated.loose
    assert "py.binop.add" not in generated.loose
