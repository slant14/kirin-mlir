"""MLIR generic form, read back into a plain tree: the first stage of the import.

Issue 06. The import mirrors the export's two stages. This module turns text
into a tree of operations, regions, blocks, attributes and types that knows
nothing about Kirin; the second stage turns that tree into Kirin IR. Keeping
them apart means the grammar is tested against real ``mlir-opt`` without a
single Kirin statement, and the Kirin side never sees text.

MLIR's Python bindings are not on PyPI and xDSL is never a runtime dependency
(``CLAUDE.md``), so the parser is written here, as recursive descent over the
characters. Generic form is what keeps it small: every operation is spelled
the same way (ADR 0002, decision 4)::

    %r = "dialect.op"(%operands) [^successors] <{properties}> ({regions}) {attributes} : (types) -> (types)

**What is accepted** is generic form as two writers produce it: this
package's emitter, and ``mlir-opt --mlir-print-op-generic``. They differ in
spelling, not in meaning, and the tests parse both and require the same tree.
``mlir-opt`` writes ``1.000000e-01`` for ``0.1``, ``-> T`` for ``-> (T)``,
``\\22`` for ``\\"``, ``dense<7>`` for three equal bytes and ``dense<"0x...">``
for many, sorts each dictionary, writes a unit entry as a bare key, drops
``: i64`` and ``: f64`` inside arrays, where they are the defaults, renumbers
every value and block, and adds ``// pred:`` comments.

**Everything else is refused** with its line and column, never skipped:
custom assembly (``module {``), aliases (``#map = ...``), locations
(``loc(...)``), dialect attributes (``#foo.bar``), and builtin attributes and
types the exporter never writes (affine maps, ``memref``, float ``dense``,
...). Text from another tool must be printed with ``--mlir-print-op-generic``
and without ``--mlir-print-debuginfo``. Strings must be UTF-8. Every string
the exporter writes is, because it writes ``bytes`` as ``dense`` i8, never as
a string.

The tree keeps MLIR's meaning, not its spelling. An integer without a type is
an ``i64`` and a float an ``f64``, as MLIR reads them; a splat or hex
``dense`` is expanded to its elements; a float compares by its bits, so
``-0.0`` and ``0.0`` differ and a NaN equals itself. Value and block names are
kept as written (``%5#1`` is the second result of ``%5:2``), for the import
to resolve: whether each is defined, and only once, is its check, not this
module's.
"""

from __future__ import annotations

import re
import struct
from bisect import bisect_right
from dataclasses import field, dataclass

from .errors import MLIRImportError

# -- types --------------------------------------------------------------------


@dataclass(frozen=True)
class IntegerType:
    """``i64``, ``si8``, ``ui32``: a width and MLIR's signedness prefix."""

    width: int
    signedness: str = ""


@dataclass(frozen=True)
class FloatType:
    """``f16``, ``bf16``, ``f32`` or ``f64``, by name."""

    name: str


@dataclass(frozen=True)
class IndexType:
    """``index``."""


@dataclass(frozen=True)
class NoneType:
    """``none``."""


@dataclass(frozen=True)
class TensorType:
    """``tensor<3xi8>``: a static shape and an element type."""

    shape: tuple[int, ...]
    element: Type


@dataclass(frozen=True)
class FunctionType:
    """``(T, U) -> (V)``, as an operation's signature or as an attribute."""

    inputs: tuple[Type, ...]
    results: tuple[Type, ...]


@dataclass(frozen=True)
class DialectType:
    """``!kirin.pyclass<"builtins.int">``: dialect, name and parameters.

    The parameters are attributes, as every ``!kirin.*`` type's are
    (:data:`kirin_mlir.types.TYPE_SCHEMA`); a type in a parameter is a
    :class:`TypeAttr`.
    """

    dialect: str
    name: str
    params: tuple[Attribute, ...] = ()


Type = (
    IntegerType
    | FloatType
    | IndexType
    | NoneType
    | TensorType
    | FunctionType
    | DialectType
)

I64 = IntegerType(64)
F64 = FloatType("f64")

# -- attributes ---------------------------------------------------------------


@dataclass(frozen=True)
class UnitAttr:
    """``unit``, or a dictionary key written without ``= value``."""


@dataclass(frozen=True)
class BoolAttr:
    """``true`` or ``false``."""

    value: bool


@dataclass(frozen=True)
class IntegerAttr:
    """``42 : i64``. Without ``: type``, MLIR reads an integer as an ``i64``."""

    value: int
    type: Type = I64


@dataclass(frozen=True, eq=False)
class FloatAttr:
    """``0.1 : f64``. Without ``: type``, MLIR reads a float as an ``f64``.

    Compared by bits, not by ``==``, so that ``-0.0`` is not ``0.0`` and a NaN
    is equal to the same NaN: two spellings of one attribute must compare
    equal, and two attributes must not.
    """

    value: float
    type: Type = F64

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FloatAttr)
            and self.type == other.type
            and _bits(self.value) == _bits(other.value)
        )

    def __hash__(self) -> int:
        return hash((self.type, _bits(self.value)))


@dataclass(frozen=True)
class StringAttr:
    """``"text"``, unescaped."""

    value: str


@dataclass(frozen=True)
class ArrayAttr:
    """``[a, b]``."""

    elements: tuple[Attribute, ...]


@dataclass(frozen=True)
class DictionaryAttr:
    """``{key = value, flag}``. A bare key holds :class:`UnitAttr`."""

    entries: dict[str, Attribute]


@dataclass(frozen=True)
class SymbolRefAttr:
    """``@callee``, or ``@outer::@inner`` with ``nested = ("inner",)``."""

    root: str
    nested: tuple[str, ...] = ()


@dataclass(frozen=True)
class DenseElementsAttr:
    """``dense<[0, 1, -1]> : tensor<3xi8>``, a splat or hex form expanded.

    Integer elements only: the exporter writes ``dense`` for ``bytes`` alone.
    """

    values: tuple[int, ...]
    type: TensorType


@dataclass(frozen=True)
class DenseArrayAttr:
    """``array<i32: 1, 1, 0>``, as in ``operandSegmentSizes``. Integers only."""

    element: Type
    values: tuple[int, ...]


@dataclass(frozen=True)
class TypeAttr:
    """A type where an attribute is expected, such as ``signature``'s."""

    type: Type


Attribute = (
    UnitAttr
    | BoolAttr
    | IntegerAttr
    | FloatAttr
    | StringAttr
    | ArrayAttr
    | DictionaryAttr
    | SymbolRefAttr
    | DenseElementsAttr
    | DenseArrayAttr
    | TypeAttr
)

# -- operations ---------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """One operation, in generic form's fields.

    Args:
        name: the quoted name, such as ``"kirin_py_binop.add"``.
        results: the result names, one per result: ``%5:2`` gives
            ``("%5#0", "%5#1")``.
        operands: the operand names, as used: ``%x``, ``%5#1``.
        successors: block labels, such as ``"^bb1"``.
        properties: ``<{...}>``, empty unless the text has it.
        regions: in order.
        attributes: the dictionary.
        operand_types: one per operand.
        result_types: one per result.
        line: where the operation starts, for the import's errors. Not compared.
    """

    name: str
    results: tuple[str, ...]
    operands: tuple[str, ...]
    successors: tuple[str, ...]
    properties: dict[str, Attribute]
    regions: tuple[Region, ...]
    attributes: dict[str, Attribute]
    operand_types: tuple[Type, ...]
    result_types: tuple[Type, ...]
    line: int = field(default=0, compare=False)


@dataclass(frozen=True)
class Block:
    """A label (``None`` for an unlabelled entry block), arguments, operations."""

    label: str | None
    arguments: tuple[tuple[str, Type], ...]
    operations: tuple[Operation, ...]
    line: int = field(default=0, compare=False)


@dataclass(frozen=True)
class Region:
    """``{...}``: its blocks, entry first."""

    blocks: tuple[Block, ...]


# -- entry points -------------------------------------------------------------


def parse_module(text: str) -> Operation:
    """The one ``"builtin.module"`` operation that ``text`` holds.

    Raises:
        MLIRImportError: anything outside the subset in this module's
            docstring, with the line and column where it starts.
    """
    parser = _Parser(text)
    parser.skip()
    if _ALIAS_DEFINITION.match(text, parser.pos):
        raise parser.error(
            "an alias definition is not read; the exporter writes none, and "
            "mlir-opt writes them only for locations and builtin attributes it "
            "never prints for this package's output"
        )
    start = parser.pos
    module = parser.operation()
    if module.name != "builtin.module":
        raise parser.error(
            f'expected one "builtin.module" operation, found "{module.name}"', start
        )
    parser.end("the module")
    return module


def parse_attribute(text: str) -> Attribute:
    """One attribute, such as ``dense<7> : tensor<3xi8>``."""
    parser = _Parser(text)
    attribute = parser.attribute()
    parser.end("the attribute")
    return attribute


def parse_type(text: str) -> Type:
    """One type, such as ``!kirin.pyclass<"builtins.int">``."""
    parser = _Parser(text)
    type_ = parser.type()
    parser.end("the type")
    return type_


# -- the parser ---------------------------------------------------------------

_SPACE = re.compile(r"\s*")
_SUFFIX_ID = r"(?:[0-9]+|[A-Za-z_$.\-][\w$.\-]*)"
_VALUE_DEF = re.compile("%" + _SUFFIX_ID)
_VALUE_USE = re.compile("%" + _SUFFIX_ID + r"(?:#[0-9]+)?")
_BLOCK_ID = re.compile(r"\^" + _SUFFIX_ID)
_BARE_ID = re.compile(r"[A-Za-z_][\w$.]*")
_DIALECT_NAME = re.compile(r"[A-Za-z_][\w$]*(?:\.[\w$.]+)?")
_ALIAS_DEFINITION = re.compile(r"[#!][\w$.]+\s*=")
_HEX = re.compile(r"-?0x[0-9A-Fa-f]+")
_FLOAT = re.compile(r"[-+]?[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?")
_INTEGER = re.compile(r"[-+]?[0-9]+")
_KEYWORD = re.compile(r"[A-Za-z_][\w$]*")
_INTEGER_TYPE = re.compile(r"([su]?)i([1-9][0-9]*)")
_FLOAT_TYPES = frozenset({"f16", "bf16", "f32", "f64"})
_DIMENSIONS = re.compile(r"(?:[0-9]+x)*")
_STRING_RUN = re.compile(r'[^"\\\n]+')
_HEX_PAIR = re.compile(r"[0-9A-Fa-f]{2}")
_ESCAPES = {'"': b'"', "\\": b"\\", "n": b"\n", "t": b"\t"}

#: Builtin attributes and types MLIR has that the exporter never writes, named
#: in the refusal so that it says what was found rather than "unexpected".
_NOT_WRITTEN = frozenset(
    {
        "affine_map",
        "affine_set",
        "complex",
        "dense_resource",
        "distinct",
        "loc",
        "memref",
        "opaque",
        "sparse",
        "strided",
        "tuple",
        "vector",
    }
)


class _Parser:
    """Recursive descent over ``text``, one method per production."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", text)]

    # -- position and errors --------------------------------------------------

    def line(self, pos: int) -> int:
        return bisect_right(self._line_starts, pos)

    def error(self, message: str, pos: int | None = None) -> MLIRImportError:
        pos = self.pos if pos is None else pos
        line = self.line(pos)
        column = pos - self._line_starts[line - 1] + 1
        return MLIRImportError(message, line=line, column=column)

    def found(self) -> str:
        if self.pos >= len(self.text):
            return "the end of the input"
        return repr(self.text[self.pos : self.pos + 24].split("\n", 1)[0])

    def skip(self) -> None:
        """Whitespace, and ``//`` comments, which ``mlir-opt`` writes."""
        while True:
            self.pos = _SPACE.match(self.text, self.pos).end()
            if not self.text.startswith("//", self.pos):
                return
            end = self.text.find("\n", self.pos)
            self.pos = len(self.text) if end < 0 else end

    def peek(self, token: str) -> bool:
        self.skip()
        return self.text.startswith(token, self.pos)

    def accept(self, token: str) -> bool:
        if self.peek(token):
            self.pos += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.accept(token):
            raise self.error(f"expected {token!r}, found {self.found()}")

    def match(self, pattern: re.Pattern[str]) -> str | None:
        self.skip()
        m = pattern.match(self.text, self.pos)
        if m is None:
            return None
        self.pos = m.end()
        return m.group()

    def require(self, pattern: re.Pattern[str], what: str) -> str:
        text = self.match(pattern)
        if text is None:
            raise self.error(f"expected {what}, found {self.found()}")
        return text

    def end(self, what: str) -> None:
        self.skip()
        if self.pos != len(self.text):
            raise self.error(
                f"expected the end of the input after {what}, found {self.found()}"
            )

    # -- operations -----------------------------------------------------------

    def operation(self) -> Operation:
        self.skip()
        start = self.pos
        results = self.result_list()
        if results:
            self.expect("=")
        if not self.peek('"'):
            raise self.error(
                f"expected an operation in generic form, a quoted name such as "
                f'"dialect.op", found {self.found()}; custom assembly is not '
                "read, so print with mlir-opt --mlir-print-op-generic"
            )
        name = self.string()
        self.expect("(")
        operands = self.delimited(self.value_use, ")")
        successors = self.successors() if self.peek("[") else ()
        properties = self.properties() if self.peek("<") else {}
        regions: tuple[Region, ...] = ()
        if self.accept("("):
            regions = self.delimited(self.region, ")")
        attributes = self.dictionary() if self.peek("{") else {}
        self.expect(":")
        signature = self.function_type()
        if self.peek("loc("):
            raise self.error(
                "locations are not read; print without --mlir-print-debuginfo"
            )
        if len(signature.inputs) != len(operands):
            raise self.error(
                f'"{name}" has {len(operands)} operands but '
                f"{len(signature.inputs)} operand types",
                start,
            )
        if len(signature.results) != len(results):
            raise self.error(
                f'"{name}" has {len(results)} results but '
                f"{len(signature.results)} result types",
                start,
            )
        return Operation(
            name,
            results,
            operands,
            successors,
            properties,
            regions,
            attributes,
            signature.inputs,
            signature.results,
            self.line(start),
        )

    def result_list(self) -> tuple[str, ...]:
        names: list[str] = []
        while self.peek("%"):
            name = self.require(_VALUE_DEF, "a result name")
            if self.accept(":"):
                count = int(self.require(_INTEGER, "a result count"))
                names += [f"{name}#{i}" for i in range(count)]
            else:
                names.append(name)
            if not self.accept(","):
                break
        return tuple(names)

    def value_use(self) -> str:
        return self.require(_VALUE_USE, "a value such as %0")

    def successors(self) -> tuple[str, ...]:
        self.expect("[")

        def successor() -> str:
            label = self.require(_BLOCK_ID, "a block label such as ^bb1")
            if self.peek("("):
                raise self.error(
                    "successor operands in brackets are not read; Kirin passes "
                    "them as operands, with operandSegmentSizes"
                )
            return label

        return self.delimited(successor, "]")

    def properties(self) -> dict[str, Attribute]:
        self.expect("<")
        entries = self.dictionary()
        self.expect(">")
        return entries

    def region(self) -> Region:
        self.expect("{")
        blocks: list[Block] = []
        if not self.peek("^") and not self.peek("}"):
            blocks.append(self.block_body(None, (), self.pos))
        while not self.accept("}"):
            blocks.append(self.block())
        return Region(tuple(blocks))

    def block(self) -> Block:
        self.skip()
        start = self.pos
        label = self.require(_BLOCK_ID, "a block label such as ^bb0")
        arguments: tuple[tuple[str, Type], ...] = ()
        if self.accept("("):

            def argument() -> tuple[str, Type]:
                name = self.require(_VALUE_DEF, "a block argument such as %arg0")
                self.expect(":")
                return name, self.type()

            arguments = self.delimited(argument, ")")
        self.expect(":")
        return self.block_body(label, arguments, start)

    def block_body(
        self, label: str | None, arguments: tuple[tuple[str, Type], ...], start: int
    ) -> Block:
        operations: list[Operation] = []
        while not self.peek("^") and not self.peek("}"):
            if self.pos >= len(self.text):
                raise self.error("the region is not closed: expected '}'")
            operations.append(self.operation())
        return Block(label, arguments, tuple(operations), self.line(start))

    def delimited(self, item, close: str) -> tuple:
        """Comma-separated ``item()`` up to ``close``, which may come at once."""
        items = []
        if self.accept(close):
            return ()
        while True:
            items.append(item())
            if self.accept(close):
                return tuple(items)
            self.expect(",")

    # -- types ----------------------------------------------------------------

    def type(self) -> Type:
        self.skip()
        start = self.pos
        if self.peek("!"):
            return self.dialect_type()
        if self.peek("("):
            return self.function_type()
        keyword = self.match(_KEYWORD)
        if keyword is None:
            raise self.error(f"expected a type, found {self.found()}")
        if m := _INTEGER_TYPE.fullmatch(keyword):
            return IntegerType(int(m.group(2)), m.group(1))
        if keyword in _FLOAT_TYPES:
            return FloatType(keyword)
        if keyword == "index":
            return IndexType()
        if keyword == "none":
            return NoneType()
        if keyword == "tensor":
            return self.tensor_type()
        if keyword in _NOT_WRITTEN:
            raise self.error(
                f"the builtin type {keyword} is not one the exporter writes", start
            )
        raise self.error(f"expected a type, found {keyword!r}", start)

    def dialect_type(self) -> DialectType:
        self.skip()
        start = self.pos
        self.expect("!")
        m = _DIALECT_NAME.match(self.text, self.pos)
        if m is None or "." not in m.group():
            raise self.error(
                "a type alias or an opaque dialect type (`!name` with no "
                "`.mnemonic`) is not read; the exporter writes neither",
                start,
            )
        self.pos = m.end()
        dialect, _, name = m.group().partition(".")
        params: tuple[Attribute, ...] = ()
        if self.text.startswith("<", self.pos):
            self.pos += 1
            params = self.delimited(self.attribute, ">")
        return DialectType(dialect, name, params)

    def function_type(self) -> FunctionType:
        self.expect("(")
        inputs = self.delimited(self.type, ")")
        self.expect("->")
        if self.accept("("):
            results = self.delimited(self.type, ")")
        else:
            results = (self.type(),)
        return FunctionType(inputs, results)

    def tensor_type(self) -> TensorType:
        if not self.text.startswith("<", self.pos):
            raise self.error(f"expected '<' after tensor, found {self.found()}")
        dims = _DIMENSIONS.match(self.text, self.pos + 1)
        self.pos = dims.end()
        shape = tuple(int(d) for d in dims.group().split("x")[:-1])
        element = self.type()
        self.expect(">")
        return TensorType(shape, element)

    # -- attributes -----------------------------------------------------------

    def attribute(self) -> Attribute:
        self.skip()
        start = self.pos
        if self.peek('"'):
            value = self.string()
            if self.peek(":"):
                raise self.error(
                    "a typed string attribute is not one the exporter writes"
                )
            return StringAttr(value)
        if self.accept("["):
            return ArrayAttr(self.delimited(self.attribute, "]"))
        if self.peek("{"):
            return DictionaryAttr(self.dictionary())
        if self.peek("@"):
            return self.symbol_ref()
        if self.peek("#"):
            raise self.error(
                "`#...` is an attribute alias or a dialect attribute; the "
                "exporter writes neither"
            )
        if self.peek("!") or self.peek("("):
            return TypeAttr(self.type())
        if (number := self.number()) is not None:
            return number
        keyword = self.match(_KEYWORD)
        if keyword is None:
            raise self.error(f"expected an attribute, found {self.found()}")
        if keyword == "unit":
            return UnitAttr()
        if keyword in ("true", "false"):
            return BoolAttr(keyword == "true")
        if keyword == "dense":
            return self.dense(start)
        if keyword == "array":
            return self.dense_array()
        if keyword in _NOT_WRITTEN:
            raise self.error(
                f"{keyword} is not an attribute the exporter writes", start
            )
        # Not an attribute keyword, so a builtin type in attribute position.
        self.pos = start
        return TypeAttr(self.type())

    def number(self) -> IntegerAttr | FloatAttr | None:
        """A number, with its optional ``: type``; ``None`` if there is none."""
        self.skip()
        start = self.pos
        if (text := self.match(_HEX)) is not None:
            type_ = self.number_type(I64)
            if isinstance(type_, FloatType):
                return FloatAttr(self.float_bits(text, type_, start), type_)
            return IntegerAttr(int(text, 16), type_)
        if (text := self.match(_FLOAT)) is not None:
            type_ = self.number_type(F64)
            if not isinstance(type_, FloatType):
                raise self.error(
                    f"the float {text} has the non-float type {type_}", start
                )
            return FloatAttr(float(text), type_)
        if (text := self.match(_INTEGER)) is not None:
            type_ = self.number_type(I64)
            if isinstance(type_, FloatType):
                return FloatAttr(float(int(text)), type_)
            return IntegerAttr(int(text), type_)
        return None

    def number_type(self, default: Type) -> Type:
        return self.type() if self.accept(":") else default

    def float_bits(self, text: str, type_: FloatType, start: int) -> float:
        """A hex float: the IEEE-754 bit pattern, as the exporter writes NaN."""
        formats = {"f64": ("<Q", "<d"), "f32": ("<I", "<f")}
        if text.startswith("-") or type_.name not in formats:
            raise self.error(f"a hex {type_.name} literal {text} is not read", start)
        bits, real = formats[type_.name]
        try:
            return struct.unpack(real, struct.pack(bits, int(text, 16)))[0]
        except struct.error:
            raise self.error(f"{text} is too wide for {type_.name}", start) from None

    def dense(self, start: int) -> DenseElementsAttr:
        if not self.text.startswith("<", self.pos):
            raise self.error(f"expected '<' after dense, found {self.found()}")
        self.pos += 1
        hex_text: str | None = None
        splat: int | None = None
        values: tuple[int, ...] = ()
        if self.peek('"'):
            hex_text = self.string()
            self.expect(">")
        elif self.accept("["):
            values = self.delimited(self.dense_element, "]")
            self.expect(">")
        elif not self.accept(">"):
            splat = self.dense_element()
            self.expect(">")
        self.expect(":")
        type_ = self.type()
        if not isinstance(type_, TensorType) or not isinstance(
            type_.element, IntegerType
        ):
            raise self.error(
                "only integer dense elements are read; the exporter writes "
                "dense for bytes alone",
                start,
            )
        count = 1
        for dim in type_.shape:
            count *= dim
        if hex_text is not None:
            values = self.dense_hex(hex_text, type_.element, count, start)
        elif splat is not None:
            values = (splat,) * count
        if len(values) != count:
            raise self.error(
                f"{len(values)} dense elements for a {count}-element tensor", start
            )
        return DenseElementsAttr(values, type_)

    def dense_element(self) -> int:
        if self.match(_FLOAT) is not None:
            raise self.error(
                "only integer dense elements are read; the exporter writes "
                "dense for bytes alone"
            )
        if (text := self.match(_INTEGER)) is None:
            raise self.error(f"expected an integer dense element, found {self.found()}")
        if self.peek("["):
            raise self.error("nested dense elements are not read")
        return int(text)

    def dense_hex(
        self, text: str, element: IntegerType, count: int, start: int
    ) -> tuple[int, ...]:
        """``dense<"0x...">``: little-endian elements, one splat if only one."""
        size, rest = divmod(element.width, 8)
        if not text.startswith("0x") or rest or size == 0:
            raise self.error(f"a hex dense of {element} is not read", start)
        try:
            data = bytes.fromhex(text[2:])
        except ValueError:
            raise self.error(f"{text!r} is not hex", start) from None
        signed = element.signedness != "u"
        values = tuple(
            int.from_bytes(data[i : i + size], "little", signed=signed)
            for i in range(0, len(data), size)
        )
        return values * count if len(values) == 1 else values

    def dense_array(self) -> DenseArrayAttr:
        if not self.text.startswith("<", self.pos):
            raise self.error(f"expected '<' after array, found {self.found()}")
        self.pos += 1
        element = self.type()
        if not isinstance(element, IntegerType):
            raise self.error(f"only integer dense arrays are read, not {element}")
        values: tuple[int, ...] = ()
        if self.accept(":"):
            values = self.delimited(
                lambda: int(self.require(_INTEGER, "an integer")), ">"
            )
        else:
            self.expect(">")
        return DenseArrayAttr(element, values)

    def dictionary(self) -> dict[str, Attribute]:
        self.expect("{")
        entries: dict[str, Attribute] = {}

        def entry() -> None:
            self.skip()
            start = self.pos
            key = (
                self.string()
                if self.peek('"')
                else self.require(_BARE_ID, "an attribute name")
            )
            if key in entries:
                raise self.error(f"attribute {key!r} appears twice", start)
            entries[key] = self.attribute() if self.accept("=") else UnitAttr()

        self.delimited(entry, "}")
        return entries

    def symbol_ref(self) -> SymbolRefAttr:
        names: list[str] = []
        while True:
            self.expect("@")
            if self.text.startswith('"', self.pos):
                names.append(self.string())
            else:
                names.append(self.require(_BARE_ID, "a symbol name"))
            if not self.accept("::"):
                return SymbolRefAttr(names[0], tuple(names[1:]))

    def string(self) -> str:
        """A string literal, unescaped: ``\\"``, ``\\\\``, ``\\n``, ``\\t``, ``\\XX``."""
        self.skip()
        start = self.pos
        if not self.text.startswith('"', start):
            raise self.error(f"expected a string, found {self.found()}")
        out = bytearray()
        i = start + 1
        while True:
            if run := _STRING_RUN.match(self.text, i):
                out += run.group().encode("utf-8")
                i = run.end()
            if i >= len(self.text) or self.text[i] == "\n":
                raise self.error("the string is not closed", start)
            if self.text[i] == '"':
                break
            escape = self.text[i + 1 : i + 2]
            if escape in _ESCAPES:
                out += _ESCAPES[escape]
                i += 2
            elif _HEX_PAIR.fullmatch(self.text[i + 1 : i + 3]):
                out.append(int(self.text[i + 1 : i + 3], 16))
                i += 3
            else:
                raise self.error(f"unknown escape {self.text[i : i + 2]!r}", i)
        self.pos = i + 1
        try:
            return out.decode("utf-8")
        except UnicodeDecodeError:
            raise self.error(
                "the string is not UTF-8; the exporter writes text as UTF-8 and "
                "bytes as dense i8",
                start,
            ) from None


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)
