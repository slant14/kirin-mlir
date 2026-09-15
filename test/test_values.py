"""Python payloads to builtin MLIR attributes."""

import math
import struct
from enum import IntEnum
from typing import NamedTuple

import pytest

from kirin_mlir import MLIRExportError, escape_string, emit_python_value


def test_none_is_a_unit_attribute():
    assert emit_python_value(None) == "unit"


def test_bool_is_checked_before_int():
    """``bool`` subclasses ``int``; ``True`` must not become ``1 : i64``."""
    assert emit_python_value(True) == "true"
    assert emit_python_value(False) == "false"
    assert emit_python_value(1) == "1 : i64"


def test_int_carries_an_explicit_width():
    assert emit_python_value(0) == "0 : i64"
    assert emit_python_value(-7) == "-7 : i64"
    assert emit_python_value(2**63 - 1) == f"{2**63 - 1} : i64"


def test_int_wider_than_i64_is_refused_not_truncated():
    with pytest.raises(MLIRExportError, match="does not fit in i64"):
        emit_python_value(2**63)
    with pytest.raises(MLIRExportError, match="does not fit in i64"):
        emit_python_value(-(2**63) - 1)


def test_float_round_trips_exactly():
    for value in (0.0, -0.0, 1.5, 1e300, 1e-300, math.pi, 3.0):
        text = emit_python_value(value)
        mantissa = text.removesuffix(" : f64")
        assert float(mantissa) == value


def test_float_always_has_a_decimal_point_in_the_mantissa():
    """``1e+300`` would lex as the integer 1 followed by an identifier."""
    assert emit_python_value(1e300) == "1.0e+300 : f64"
    assert emit_python_value(3.0) == "3.0 : f64"


def test_non_finite_floats_use_the_ieee_bit_pattern():
    for value in (math.inf, -math.inf, math.nan):
        text = emit_python_value(value)
        assert text.startswith("0x") and text.endswith(" : f64")
        bits = int(text[2:].removesuffix(" : f64"), 16)
        (recovered,) = struct.unpack(">d", struct.pack(">Q", bits))
        assert recovered == value or (math.isnan(recovered) and math.isnan(value))


def test_str_is_escaped_the_way_mlir_escapes():
    assert emit_python_value("plain") == '"plain"'
    assert emit_python_value('say "hi"') == '"say \\"hi\\""'
    assert emit_python_value("back\\slash") == '"back\\\\slash"'
    assert emit_python_value("line\nbreak") == '"line\\0Abreak"'


def test_str_is_escaped_as_utf8_bytes():
    assert escape_string("é") == "\\C3\\A9"


def test_sequences_become_array_attributes():
    assert emit_python_value(()) == "[]"
    assert emit_python_value(("x",)) == '["x"]'
    assert emit_python_value([1, 2]) == "[1 : i64, 2 : i64]"
    assert emit_python_value((1, ("a", None))) == '[1 : i64, ["a", unit]]'


def test_dicts_become_dictionary_attributes_with_quoted_keys():
    assert emit_python_value({}) == "{}"
    assert emit_python_value({"a": 1}) == '{"a" = 1 : i64}'
    assert emit_python_value({"has space": True}) == '{"has space" = true}'


def test_dicts_with_non_str_keys_are_refused():
    with pytest.raises(MLIRExportError, match="non-str keys"):
        emit_python_value({1: "a"})


def test_bytes_stay_distinguishable_from_str():
    assert emit_python_value(b"") == "dense<> : tensor<0xi8>"
    assert emit_python_value(b"\x01\x02") == "dense<[1, 2]> : tensor<2xi8>"
    # signless i8 is parsed in the signed range; the bits are unchanged
    assert emit_python_value(b"\xff") == "dense<[-1]> : tensor<1xi8>"


def test_a_class_payload_becomes_a_type_attribute():
    assert emit_python_value(int) == '!kirin.pyclass<"builtins.int">'


@pytest.mark.parametrize(
    "value, reason",
    [
        ({1, 2}, "assert an element order"),
        (frozenset({1}), "assert an element order"),
        (range(3), "start/stop/step"),
        (slice(1, 2), "start/stop/step"),
        (1 + 2j, "no builtin complex attribute"),
    ],
)
def test_payloads_without_an_honest_counterpart_are_refused(value, reason):
    with pytest.raises(MLIRExportError, match=reason):
        emit_python_value(value)


def test_an_arbitrary_object_is_refused():
    class Whatever:
        pass

    with pytest.raises(MLIRExportError, match="SUPPORTED_PYTHON_TYPES"):
        emit_python_value(Whatever())


def test_subclasses_of_builtins_are_refused_not_flattened():
    """A NamedTuple is not a tuple, and an IntEnum is not an int."""

    class Point(NamedTuple):
        x: int

    class Colour(IntEnum):
        RED = 1

    with pytest.raises(MLIRExportError):
        emit_python_value(Point(1))
    with pytest.raises(MLIRExportError):
        emit_python_value(Colour.RED)


def test_refusal_propagates_out_of_a_container():
    with pytest.raises(MLIRExportError, match="start/stop/step"):
        emit_python_value([1, range(2)])
