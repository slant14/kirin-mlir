"""The import direction, MLIR back to Kirin: issue 06, steps 2 to 6.

Built up a step at a time. Step 2 maps every operation the emitter writes
back to the statement class it was emitted for, through the same name table
the export used, so case and dots come back from Kirin's registries.
"""

from collections import Counter

import pytest
from test_irdl import CORPUS
from test_parse import operations

from kirin_mlir import NameTable, emit_mlir, parse_module, prepare_for_mlir


def ids(kernel) -> str:
    return kernel.sym_name


@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_every_parsed_operation_names_the_statement_it_came_from(kernel):
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
