"""Every spelling this package produces, handed to real ``mlir-opt``.

Skipped when ``mlir-opt`` is absent. That is not a silent pass: the skip reason
says what to install, and nothing else in the suite claims to have checked MLIR
syntax. See ``test/oracle.py`` for what this can and cannot prove at issue 02.
"""

import pytest
from kirin import types
from oracle import (
    MLIR_OPT,
    SKIP_REASON,
    parses,
    module_with_types,
    module_with_attributes,
)
from kirin.ir.attrs.types import Union, Vararg, Generic, Literal, FunctionType

from kirin_mlir import emit_type, emit_python_value

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


def test_the_oracle_actually_rejects_something():
    """Guard against a green suite that is only ever handed valid input."""
    result = parses('"builtin.module"() ({ this is not mlir }) : () -> ()\n')
    assert result.returncode != 0
