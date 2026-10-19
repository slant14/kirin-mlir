"""Exporting the stim sample programs of issue 05.

Step 5's survey found that the exporter refuses nothing in these programs, but
that Kirin before 0.22.17 typed some of their values ``Bottom``: its fallback
type inference paired fields with values by position, so a group of arguments
shifted every later value onto the wrong field's type. ``DETECTOR(0, 0)``'s
second coordinate met the ``RecordResult`` type of ``targets``. MLIR accepts
``!kirin.bottom`` as readily as the right type, so no ``mlir-opt`` check here
would notice, and the first test looks at the Kirin types directly.

The next checks give the exported text to real ``mlir-opt`` without the stim
IRDL (step 7). Every ``kirin_stim_*`` operation is then unregistered, so they
prove the text is well-formed MLIR and that no statement is lost on the way,
not that the operations are shaped right.

The last checks load the checked-in ``kirin_stim.irdl.mlir`` (steps 8-9),
which registers four of stim's five dialects, so every gate, measurement,
reset and noise channel is verified. ``stim.cf`` and Kirin ``func`` stay
unregistered (ADR 0002). Skipped without the ``stim`` extra, and the
``mlir-opt`` checks without ``mlir-opt``.
"""

import re
from pathlib import Path
from collections import Counter

import pytest
from kirin import ir, types
from oracle import MLIR_OPT, SKIP_REASON, mlir_opt
from test_irdl import hybrid
from test_parse import canonical, operations
from test_import import (
    rebuilt,
    assert_rebuilt,
    assert_kernels_come_back,
    assert_types_and_attributes_read_back,
)

from kirin_mlir import (
    NameTable,
    emit_mlir,
    import_mlir,
    parse_module,
    generate_irdl,
    prepare_for_mlir,
)
from kirin_mlir.cli import main

pytest.importorskip("bloqade.stim", reason="needs the stim extra")

from stim_kernels import EVERY, bell, dialects
from bloqade.stim.dialects import gate

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)

# Checked in, so an export can be verified without bloqade installed, and a
# change to bloqade's statements shows up as a diff of this file.
STIM_IRDL = Path(__file__).parents[1] / "kirin_stim.irdl.mlir"
REGISTRABLE = ["stim.aux", "stim.collapse", "stim.gate", "stim.noise"]
REGENERATE = ["irdl", *REGISTRABLE, "--package", "bloqade.stim", "-o"]


def ids(kernel: ir.Method) -> str:
    return kernel.sym_name


@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_no_exported_value_is_typed_bottom(kernel):
    """Bottom claims a value never exists; every value here is used."""
    module = prepare_for_mlir(kernel)
    bottom = [
        f"{stmt.dialect.name}.{stmt.name} in {method.sym_name}"
        for method in module.methods
        for stmt in method.callable_region.walk()
        for value in stmt.results
        if value.type == types.Bottom
    ]
    assert bottom == []


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_program_round_trips(kernel):
    """Unregistered, so parsed and printed back, not verified."""
    text = emit_mlir(prepare_for_mlir(kernel))
    result = mlir_opt(text, "--allow-unregistered-dialect", "--verify-roundtrip")
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_statement_reaches_mlir_under_its_own_name(kernel):
    """MLIR parses one operation per Kirin statement, named by the name table.

    Counted from ``mlir-opt``'s own generic printout, so this checks what MLIR
    read, not what the emitter meant to write. ``Method.print()`` shows these
    bodies as empty (issue 05, step 4), so a dropped statement would not be
    visible by eye.
    """
    module = prepare_for_mlir(kernel)
    result = mlir_opt(
        emit_mlir(module), "--allow-unregistered-dialect", "--mlir-print-op-generic"
    )
    assert result.returncode == 0, result.stderr
    parsed = Counter(re.findall(r'"([\w.$]+)"\(', result.stdout))
    assert parsed.pop("builtin.module") == 1

    names = NameTable.of(module.dialects)
    expected = Counter(
        names.operation(stmt)
        for method in module.methods
        for stmt in method.code.walk()
    )
    assert parsed == expected
    assert any(name.startswith("kirin_stim_") for name in parsed)


# -- verified against the stim IRDL -------------------------------------------


def test_only_stim_cf_cannot_be_registered():
    """ADR 0002 decision 2 over all of stim; the rest is what the file covers."""
    generated = generate_irdl(dialects())
    assert [str(e) for e in generated.excluded] == ["stim.cf.REPEAT has a region"]
    assert list(generated.registered) == REGISTRABLE
    # The statements are bloqade's, so the header must say which bloqade.
    assert "bloqade-circuit" in generated.text.splitlines()[1]


@needs_mlir
def test_the_checked_in_irdl_is_current(tmp_path):
    """``kirin_stim.irdl.mlir`` declares what the generator declares today.

    Compared as ``mlir-opt`` reads the two files, so the versions in the header
    comment do not count, and a changed statement, operand or attribute does.
    """
    fresh = tmp_path / STIM_IRDL.name
    assert main([*REGENERATE, str(fresh)]) == 0

    def read(path: Path) -> str:
        result = mlir_opt(path.read_text())
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert read(STIM_IRDL) == read(fresh), (
        f"{STIM_IRDL.name} is stale; regenerate it with "
        f"`kirin-mlir {' '.join(REGENERATE)} {STIM_IRDL.name}`"
    )


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_program_verifies_against_the_stim_irdl(kernel):
    """Issue 05's first criterion, with the checked-in file."""
    text = emit_mlir(prepare_for_mlir(kernel))
    result = hybrid(text, str(STIM_IRDL), "--verify-roundtrip")
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_program_reads_back_as_the_same_tree_from_mlir_opt(kernel):
    """Issue 06's parser on the stim programs, as MLIR prints them verified.

    With the stim IRDL loaded, so the registered operations and the ``REPEAT``
    region are printed by real ``mlir-opt`` and read back like our own text.
    """
    text = emit_mlir(prepare_for_mlir(kernel))
    printed = hybrid(text, str(STIM_IRDL), "--mlir-print-op-generic")
    assert printed.returncode == 0, printed.stderr
    assert canonical(parse_module(printed.stdout)) == canonical(parse_module(text))


@needs_mlir
def test_every_stim_statement_is_verified_by_mlir():
    """Issue 05's fourth criterion, after export.

    Every statement of stim's five dialects appears in some program that
    verifies against the checked-in IRDL, counted from ``mlir-opt``'s own
    printout. ``REPEAT`` is among them, parsed but not verified (ADR 0002).
    """
    parsed = set()
    for kernel in EVERY:
        result = hybrid(
            emit_mlir(prepare_for_mlir(kernel)),
            str(STIM_IRDL),
            "--mlir-print-op-generic",
        )
        assert result.returncode == 0, result.stderr
        parsed |= set(re.findall(r'"(kirin_stim_[\w.]+)"\(', result.stdout))

    names = NameTable.of(dialects())
    declared = {names.operation(stmt) for d in dialects() for stmt in d.stmts}
    assert sorted(declared - parsed) == []


def _drop_dagger(module):
    cx = next(s for s in module.entry.callable_region.walk() if isinstance(s, gate.CX))
    del cx.attributes["dagger"]
    return emit_mlir(module)


def _miscount_segments(module):
    """``bell``'s one ``CX`` has a target and a control; claim two targets."""
    text = emit_mlir(module)
    cx = re.search(r'"kirin_stim_gate\.cx".*', text).group()
    assert "array<i32: 1, 1>" in cx
    return text.replace(cx, cx.replace("array<i32: 1, 1>", "array<i32: 2, 1>"))


@needs_mlir
@pytest.mark.parametrize(
    "corrupt, error",
    [
        (_drop_dagger, 'attribute "dagger" is expected'),
        (_miscount_segments, "sum of elements in 'operandSegmentSizes'"),
    ],
    ids=["cx without dagger", "segment sizes that do not sum"],
)
def test_a_corrupted_program_is_rejected_only_because_of_the_stim_irdl(corrupt, error):
    """Issue 05's second criterion.

    The same text passes with ``--allow-unregistered-dialect`` alone, so it is
    registration that rejects it. A wrong *number* of targets is no control:
    target lists are variadic, so IRDL rightly accepts any number.
    """
    text = corrupt(prepare_for_mlir(bell))

    assert mlir_opt(text, "--allow-unregistered-dialect").returncode == 0
    result = hybrid(text, str(STIM_IRDL))
    assert result.returncode != 0
    assert error in result.stderr


# -- read back (issue 06) -------------------------------------------------------


def test_every_stim_name_maps_back_to_its_statement():
    """Issue 06 step 2, where 44 of stim's 53 names lose case or dots."""
    table = NameTable.of(bell.dialects)
    statements = [stmt for dialect in bell.dialects.data for stmt in dialect.stmts]
    assert all(table.statement(table.operation(stmt)) is stmt for stmt in statements)
    assert table.statement("kirin_stim_gate.sqrt_x") is gate.SqrtX
    assert table.statement("kirin_stim_aux.constant_int").name == "constant.int"


@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_parsed_stim_operation_names_the_statement_it_came_from(kernel):
    module = prepare_for_mlir(kernel)
    table = NameTable.of(module.dialects)
    tree = parse_module(emit_mlir(module))
    parsed = Counter(
        table.statement(op.name)
        for op in operations(tree)
        if op.name != "builtin.module"
    )
    expected = Counter(
        type(stmt) for method in module.methods for stmt in method.code.walk()
    )
    assert parsed == expected


@pytest.mark.parametrize("printed", [False, pytest.param(True, marks=needs_mlir)])
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_stim_type_and_attribute_reads_back(kernel, printed):
    """Issue 06 step 3: every ``dagger``, constant and ``REPEAT`` count.

    From our text, and from ``mlir-opt``'s printout with the stim IRDL loaded,
    so from operations real MLIR has verified.
    """
    module = prepare_for_mlir(kernel)
    text = emit_mlir(module)
    if printed:
        result = hybrid(text, str(STIM_IRDL), "--mlir-print-op-generic")
        assert result.returncode == 0, result.stderr
        text = result.stdout
    assert_types_and_attributes_read_back(module, parse_module(text))


@pytest.mark.parametrize("printed", [False, pytest.param(True, marks=needs_mlir)])
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_stim_program_is_rebuilt_as_its_prepared_copy(kernel, printed):
    """Issue 06 step 4, ``REPEAT``'s region and its terminator-less block too."""
    module = prepare_for_mlir(kernel)
    text = emit_mlir(module)
    if printed:
        result = hybrid(text, str(STIM_IRDL), "--mlir-print-op-generic")
        assert result.returncode == 0, result.stderr
        text = result.stdout
    assert_rebuilt(module, rebuilt(module, text))


@pytest.mark.parametrize("printed", [False, pytest.param(True, marks=needs_mlir)])
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_stim_program_comes_back_as_its_kernel(kernel, printed):
    """Issue 06 step 5: ``import_mlir`` on what MLIR verified."""
    module = prepare_for_mlir(kernel)
    text = emit_mlir(module)
    if printed:
        result = hybrid(text, str(STIM_IRDL), "--mlir-print-op-generic")
        assert result.returncode == 0, result.stderr
        text = result.stdout
    assert_kernels_come_back(module, import_mlir(text, kernel.dialects))
