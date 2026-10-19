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

The way back (issue 06), :func:`import_python_value`, reads a parsed
attribute as the payload it was written from. Two pairs of Python types share
a spelling, ``tuple`` and ``list``, and ``bytes`` and ``bytearray``, so it
takes the Kirin type the payload was declared with and lets that decide.
"""

from __future__ import annotations

import math
import struct
from typing import Any
from collections.abc import Callable

from kirin.ir.attrs.types import (
    Union,
    AnyType,
    Generic,
    Literal,
    PyClass,
    TypeVar,
    TypeAttribute,
)

from . import parse
from .errors import MLIRExportError, MLIRImportError

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


# -- the way back (issue 06) --------------------------------------------------


def import_python_value(attr: parse.Attribute, hint: TypeAttribute) -> Any:
    """The Python payload :func:`emit_python_value` wrote as ``attr``.

    Its inverse, for the import. The attribute says what the payload is,
    except where two Python types share one spelling: an array is a ``tuple``
    or a ``list``, and a ``dense`` i8 is ``bytes`` or a ``bytearray``.
    ``hint``, the Kirin type the payload was declared with, decides those:
    a statement field's type, or a ``Literal``'s. What it leaves open is
    refused, never guessed (issue 06, step 0, decision 8). Nothing else is
    checked against it: ``1 : i64`` under a field declared ``float`` reads
    back as the ``int`` the exporter was given.

    A ``dict`` comes back in the order the text has its entries. ``mlir-opt``
    sorts every dictionary by key, so after a pass through it the order is
    sorted. ``==`` does not see that, but iteration does.

    Raises:
        MLIRImportError: for an attribute :func:`emit_python_value` never
            writes, or an array or ``dense`` that ``hint`` does not decide.
    """
    if isinstance(attr, parse.UnitAttr):
        return None
    if isinstance(attr, parse.BoolAttr):
        return attr.value
    if isinstance(attr, parse.IntegerAttr):
        if attr.type != parse.I64:
            raise MLIRImportError(
                f"found an integer of type {attr.type}; the exporter writes every "
                "integer as an i64"
            )
        return attr.value
    if isinstance(attr, parse.FloatAttr):
        if attr.type != parse.F64:
            raise MLIRImportError(
                f"found a float of type {attr.type}; the exporter writes every "
                "float as an f64"
            )
        return attr.value
    if isinstance(attr, parse.StringAttr):
        return attr.value
    if isinstance(attr, parse.ArrayAttr):
        kind = _declared(hint, (tuple, list), "an array")
        items = _item_hints(hint, kind, len(attr.elements))
        return kind(map(import_python_value, attr.elements, items))
    if isinstance(attr, parse.DictionaryAttr):
        items = _item_hints(hint, dict, len(attr.entries))
        return {
            key: import_python_value(value, item)
            for (key, value), item in zip(attr.entries.items(), items)
        }
    if isinstance(attr, parse.DenseElementsAttr):
        bytes_type = parse.TensorType((len(attr.values),), parse.IntegerType(8))
        if attr.type != bytes_type or not all(-128 <= v < 128 for v in attr.values):
            raise MLIRImportError(
                f"found dense elements of type {attr.type}; the exporter writes "
                "dense only for bytes, as a one-dimensional tensor of i8"
            )
        kind = _declared(hint, (bytes, bytearray), "a dense i8 tensor")
        return kind(v & 0xFF for v in attr.values)
    if isinstance(attr, parse.TypeAttr):
        # The one back-edge, as on the way out: a class used as data.
        from .types import import_type

        typ = import_type(attr.type)
        if not isinstance(typ, PyClass):
            raise MLIRImportError(
                f"found the type {attr.type} as data; a type is data only as the "
                "!kirin.pyclass of a Python class"
            )
        return typ.typ
    if isinstance(attr, parse.SymbolRefAttr):
        raise MLIRImportError(
            "found a symbol reference inside a payload; the exporter writes a "
            "kernel as @symbol only as a whole attribute, never inside one"
        )
    raise MLIRImportError(
        f"found {type(attr).__name__}, which the exporter never writes as data; "
        "array<...> is written only for operandSegmentSizes"
    )


def _declared(hint: TypeAttribute, kinds: tuple[type, ...], what: str) -> type:
    """The one class of ``kinds`` that ``hint`` declares, or a refusal."""
    declared = [kind for kind in kinds if _members(hint, kind)]
    if len(declared) == 1:
        return declared[0]
    names = " or ".join(kind.__name__ for kind in kinds)
    raise MLIRImportError(
        f"{what} may be a {names}, and the declared type {hint!r} does not say "
        "which; the import reads it only against a declared type that does"
    )


def _members(hint: TypeAttribute, kind: type) -> list[PyClass | Generic]:
    """The parts of ``hint`` that declare exactly the class ``kind``.

    Exactly, because :func:`emit_python_value` dispatches on the exact type:
    a payload written as an array was a ``tuple`` or a ``list``, never a
    subclass of one.
    """
    if isinstance(hint, Union):
        return [found for member in hint.types for found in _members(member, kind)]
    if isinstance(hint, TypeVar):
        return _members(hint.bound, kind)
    if isinstance(hint, Literal):
        return _members(hint.type, kind)
    if isinstance(hint, PyClass) and hint.typ is kind:
        return [hint]
    if isinstance(hint, Generic) and hint.body.typ is kind:
        return [hint]
    return []


def _item_hints(hint: TypeAttribute, kind: type, count: int) -> list[TypeAttribute]:
    """The declared type of each of ``count`` items, or of a dict's values.

    ``tuple[int, str]`` declares each position, ``tuple[str, ...]`` and
    ``list[str]`` every item, and ``dict[str, int]`` every value. Anything
    else declares nothing about the items, and neither does a ``hint`` with
    two members of the same class, which would leave the choice to set order.
    """
    members = _members(hint, kind)
    if len(members) == 1 and isinstance(generic := members[0], Generic):
        if kind is tuple:
            rest = generic.vararg.typ if generic.vararg is not None else AnyType()
            return [
                generic.vars[i] if i < len(generic.vars) else rest for i in range(count)
            ]
        if generic.vars:
            return [generic.vars[-1]] * count
    return [AnyType()] * count
