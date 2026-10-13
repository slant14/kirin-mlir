"""Exporting the stim sample programs of issue 05.

Step 5's survey found that the exporter refuses nothing in these programs, but
that Kirin before 0.22.17 typed some of their values ``Bottom``: its fallback
type inference paired fields with values by position, so a group of arguments
shifted every later value onto the wrong field's type. ``DETECTOR(0, 0)``'s
second coordinate met the ``RecordResult`` type of ``targets``. MLIR accepts
``!kirin.bottom`` as readily as the right type, so no ``mlir-opt`` check here
would notice, and the first test looks at the Kirin types directly.

The rest check the exported text with real ``mlir-opt`` (step 7). Without the
stim IRDL every ``kirin_stim_*`` operation is unregistered, so this proves the
text is well-formed MLIR and that no statement is lost on the way, not that
the operations are shaped right; that needs the IRDL (steps 8-9). Skipped
without the ``stim`` extra, and the ``mlir-opt`` checks without ``mlir-opt``.
"""

import re
from collections import Counter

import pytest
from kirin import ir, types
from oracle import MLIR_OPT, SKIP_REASON, mlir_opt

from kirin_mlir import NameTable, emit_mlir, prepare_for_mlir

pytest.importorskip("bloqade.stim", reason="needs the stim extra")

from stim_kernels import EVERY

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)


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
