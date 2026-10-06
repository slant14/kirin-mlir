"""Every spelling this package produces, handed to real ``mlir-opt``.

Skipped when ``mlir-opt`` is absent. That is not a silent pass: the skip reason
says what to install, and nothing else in the suite claims to have checked MLIR
syntax. See ``test/oracle.py`` for what this can and cannot prove at issue 02.
"""

import collections

import pytest
from kirin import types
from oracle import (
    MLIR_OPT,
    SKIP_REASON,
    parses,
    module_with_types,
    module_with_attributes,
)
from kirin_dialects import public_dialects
from kirin.ir.attrs.types import Union, Vararg, Generic, Literal, FunctionType

from kirin_mlir import NameTable, emit_type, emit_python_value

pytestmark = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)

EVERY_TYPE = [
    types.Any,
    types.Bottom,
    types.Int,
    types.NoneType,
    Literal(3),
    Literal("text"),
    Union([types.Int, types.String]),
    Vararg(types.Any),
    types.List,
    types.Tuple,
    Generic(dict, types.String, types.Int),
    types.MethodType,
    types.MethodType[[types.Int], types.Bool],
    FunctionType((types.Int,)),
]

EVERY_PAYLOAD = [
    None,
    True,
    -1,
    2**63 - 1,
    1.5,
    1e300,
    float("inf"),
    float("nan"),
    "plain",
    'quotes " and \\ and newline \n',
    "unicode é",
    b"\x00\x7f\x80\xff",
    b"",
    (),
    (1, "a", None),
    [1, [2]],
    {},
    {"a": 1, "has space": True},
    int,
]


def test_every_type_spelling_parses():
    text = module_with_types([emit_type(t) for t in EVERY_TYPE])
    result = parses(text)
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


@pytest.mark.parametrize("payload", EVERY_PAYLOAD, ids=repr)
def test_every_payload_spelling_parses(payload):
    text = module_with_attributes({"a": emit_python_value(payload)})
    result = parses(text)
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


def test_every_kirin_name_is_a_legal_irdl_name():
    """Declare every namespace and operation in one IRDL file; MLIR must load it.

    IRDL checks names when the file is parsed, so this is real MLIR's verdict
    on every name :class:`NameTable` hands out for the whole of Kirin.
    """
    table = NameTable.of(public_dialects())
    ops = collections.defaultdict(list)
    for namespace in table.namespaces.values():
        ops[namespace]  # dialects with no statements still need a legal name
    for full in table.operations.values():
        namespace, op = full.split(".", 1)
        ops[namespace].append(op)
    text = "".join(
        f"irdl.dialect @{namespace} {{\n"
        + "".join(f"  irdl.operation @{op} {{\n  }}\n" for op in sorted(names))
        + "}\n"
        for namespace, names in sorted(ops.items())
    )
    result = parses(text)
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


def test_irdl_rejects_an_unmangled_kirin_name():
    """The negative control for the test above, and the reason names.py exists."""
    result = parses("irdl.dialect @py {\n  irdl.operation @binop.add {\n  }\n}\n")
    assert result.returncode != 0


def test_the_oracle_actually_rejects_something():
    """Guard against a green suite that is only ever handed valid input."""
    result = parses('"builtin.module"() ({ this is not mlir }) : () -> ()\n')
    assert result.returncode != 0
