"""Python payloads to builtin MLIR attributes.

This is the leaf of the mapping: it turns a raw Python object -- the ``data``
inside a :class:`kirin.ir.PyAttr`, or inside a ``types.Literal`` -- into the
text of a **builtin** MLIR attribute. It knows nothing about Kirin's IR.

It lives in its own module because both directions need it and a direct
``types <-> attrs`` cycle would otherwise be forced:

    values.py  (leaf)          emit_python_value, escape_string
      ^                ^
      |                |
    types.py         attrs.py

The one back-edge is ``type`` payloads: a Python class used as data is also a
Kirin ``PyClass`` type, so that single branch imports :mod:`kirin_mlir.types`
lazily, inside the function.

Which payloads are allowed at all is not our judgment call. Kirin's own
serialization layer already had to answer it, in
``kirin/serialization/core/supportedtypes.py`` (``SUPPORTED_PYTHON_TYPES``);
that answer is reused rather than re-derived. Of that set, the ones with no
*unambiguous* builtin MLIR counterpart are refused here, individually and with
a reason.
"""

from __future__ import annotations

import math
import struct
from typing import Any
from collections.abc import Callable

from .errors import MLIRExportError

#: MLIR's builtin ``IntegerAttr`` needs a fixed bit width. Python's ``int`` has
#: none. i64 is the carrier; anything that does not fit is refused rather than
#: truncated.
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def escape_string(text: str) -> str:
    """Escape a Python ``str`` for an MLIR string literal.

    MLIR string literals hold **bytes**, not code points, and MLIR's own
    printer escapes anything non-printable as ``\\XX`` with two uppercase hex
    digits. We encode to UTF-8 first and follow the same rule, so a string that
    contains a newline, a tab, or any non-ASCII character survives the round
    trip byte-exactly instead of depending on how a parser treats ``\\n``.
    """
    out: list[str] = []
    for byte in text.encode("utf-8"):
        if byte == 0x22:  # "
            out.append('\\"')
        elif byte == 0x5C:  # backslash
            out.append("\\\\")
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:02X}")
    return "".join(out)


def emit_python_value(value: Any) -> str:
    """Emit one Python payload as builtin MLIR attribute text.

    Raises:
        MLIRExportError: if the payload has no faithful builtin counterpart.
    """
    if value is None:
        return "unit"

    handler = _HANDLERS.get(type(value))
    if handler is not None:
        return handler(value)

    # A class object is data here, but it is also a Kirin type. Lazy import:
    # this is the one edge that would otherwise make values <-> types a cycle.
    if isinstance(value, type):
        from .types import pyclass_type

        return pyclass_type(value)

    refusal = _REFUSALS.get(type(value))
    if refusal is not None:
        raise MLIRExportError(
            f"cannot represent a Python {type(value).__name__} as an MLIR "
            f"attribute: {refusal}"
        )

    raise MLIRExportError(
        f"cannot represent a Python {type(value).__name__} as an MLIR "
        "attribute; only the exact types in Kirin's SUPPORTED_PYTHON_TYPES "
        "(kirin/serialization/core/supportedtypes.py) that have an unambiguous "
        "builtin MLIR counterpart may cross the boundary"
    )


def _emit_bool(value: bool) -> str:
    # BoolAttr, i.e. an i1 IntegerAttr. Checked before int because in Python
    # ``bool`` is a subclass of ``int`` and ``True`` would otherwise be emitted
    # as ``1 : i64``, which is a different attribute.
    return "true" if value else "false"


def _emit_int(value: int) -> str:
    if not INT64_MIN <= value <= INT64_MAX:
        raise MLIRExportError(
            f"integer {value} does not fit in i64; Python integers are "
            "arbitrary precision and MLIR's IntegerAttr is not, so this "
            "cannot be represented without changing its value"
        )
    return f"{value} : i64"


def _emit_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        # No decimal spelling of these round-trips through every parser, but
        # the IEEE-754 bit pattern does, and MLIR accepts it for FloatAttr.
        (bits,) = struct.unpack(">Q", struct.pack(">d", value))
        return f"0x{bits:016X} : f64"

    # repr() is the shortest decimal that round-trips back to the same double,
    # and MLIR's parser is correctly rounded, so this is bit-exact. The only
    # fix-up is MLIR's lexer, which needs a '.' in the mantissa: bare ``1e+300``
    # would lex as the integer 1 followed by an identifier.
    text = repr(value)
    mantissa, exponent_sep, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    text = mantissa + exponent_sep + exponent
    return f"{text} : f64"


def _emit_str(value: str) -> str:
    return f'"{escape_string(value)}"'


def _emit_bytes(value: bytes | bytearray) -> str:
    # StringAttr would also hold these bytes, but then bytes and str would be
    # indistinguishable. A dense i8 elements attribute keeps them apart. i8 is
    # signless in MLIR and parsed in the signed range, so 0x80..0xFF are
    # written as their two's-complement negatives; the bits are identical.
    if not value:
        return "dense<> : tensor<0xi8>"
    signed = [b - 256 if b > 127 else b for b in value]
    body = ", ".join(str(b) for b in signed)
    return f"dense<[{body}]> : tensor<{len(value)}xi8>"


def _emit_sequence(value: tuple | list) -> str:
    return "[" + ", ".join(emit_python_value(item) for item in value) + "]"


def _emit_dict(value: dict) -> str:
    entries: list[str] = []
    for key, item in value.items():
        if type(key) is not str:
            raise MLIRExportError(
                "cannot represent a dict with non-str keys as an MLIR "
                f"DictionaryAttr; key {key!r} is a {type(key).__name__}"
            )
        entries.append(f'"{escape_string(key)}" = {emit_python_value(item)}')
    return "{" + ", ".join(entries) + "}"


#: Exact-type dispatch, deliberately not ``isinstance``. A subclass of ``int``
#: or ``tuple`` -- a ``NamedTuple``, an ``IntEnum`` -- is not the builtin it
#: derives from, and emitting it as one would drop the very thing that made it
#: a distinct type. Refusing is visible; widening is not.
_HANDLERS: dict[type, Callable[[Any], str]] = {
    bool: _emit_bool,
    int: _emit_int,
    float: _emit_float,
    str: _emit_str,
    bytes: _emit_bytes,
    bytearray: _emit_bytes,
    tuple: _emit_sequence,
    list: _emit_sequence,
    dict: _emit_dict,
}

#: Payloads Kirin's serializer accepts but that have no honest builtin MLIR
#: spelling. Each refusal states what would have to be asserted falsely.
_REFUSALS: dict[type, str] = {
    set: (
        "an MLIR ArrayAttr is ordered and a Python set is not, so emitting one "
        "would assert an element order the program does not have"
    ),
    frozenset: (
        "an MLIR ArrayAttr is ordered and a Python frozenset is not, so "
        "emitting one would assert an element order the program does not have"
    ),
    range: (
        "there is no builtin MLIR attribute for a start/stop/step triple, and "
        "flattening it to an array would lose that it is a range"
    ),
    slice: (
        "there is no builtin MLIR attribute for a start/stop/step triple, and "
        "flattening it to an array would lose that it is a slice"
    ),
    complex: (
        "MLIR's ComplexType is a value type, not an attribute; there is no "
        "builtin complex attribute to carry this"
    ),
}
