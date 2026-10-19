"""The import direction, MLIR back to Kirin: issue 06, steps 2 to 6.

Built up a step at a time. Step 2 maps every operation the emitter writes
back to the statement class it was emitted for, through the same name table
the export used, so case and dots come back from Kirin's registries.

Step 3 reads types and attributes back. The strongest check takes every
statement of every test kernel, from our text and from ``mlir-opt``'s
printout, and requires each attribute, operand type, result type and block
argument type to import to what the prepared copy holds. The rest checks what
the corpus does not reach: each kind of type and payload, the two losses the
export makes on purpose, and the refusals.
"""

import sys
import struct
from collections import Counter

import pytest
from kirin import ir, types
from oracle import MLIR_OPT, SKIP_REASON
from test_irdl import CORPUS
from kirin.decl import info, statement
from test_parse import PAYLOADS, generic, operations
from kirin.dialects import py, func, ilist
from kirin.ir.attrs.types import (
    Union,
    Vararg,
    Generic,
    Literal,
    PyClass,
    TypeVar,
    FunctionType,
)

from kirin_mlir import (
    TYPE_SCHEMA,
    TYPE_IMPORTERS,
    NameTable,
    MLIRImportError,
    emit_mlir,
    emit_type,
    parse_type,
    import_type,
    parse_module,
    pyclass_named,
    qualified_name,
    parse_attribute,
    prepare_for_mlir,
    emit_python_value,
    import_python_value,
    import_statement_attribute,
    import_statement_attributes,
)
from kirin_mlir.emit import SEGMENT_SIZES

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)


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


# -- step 3: types and attributes back, over the corpus ----------------------


def same(a, b) -> bool:
    """``a == b``, with the same Python types all the way down.

    ``==`` alone would pass ``1`` for ``True``, ``[1]`` for ``(1,)`` and
    ``0.0`` for ``-0.0``, and fail a NaN against itself. Floats compare by
    their bits, so every one of those is caught.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack(">d", a) == struct.pack(">d", b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[key], b[key]) for key in a)
    if isinstance(a, ir.PyAttr):
        return a.type == b.type and same(a.data, b.data)
    return a == b


def block_argument_types(stmt: ir.Statement) -> list[types.TypeAttribute]:
    """The types of ``stmt``'s block arguments as the emitter writes them."""
    arguments = [
        arg for region in stmt.regions for block in region.blocks for arg in block.args
    ]
    if isinstance(stmt, func.Function):
        arguments = arguments[1:]  # %self, dropped (ADR 0001, decision 2)
    return [arg.type for arg in arguments]


def assert_types_and_attributes_read_back(module, tree) -> None:
    """Every type and attribute in ``tree`` imports to what ``module`` holds.

    The parsed operations and the prepared statements are paired in order,
    both outermost first, and each pair is checked to be the same statement
    before anything else is compared.
    """
    table = NameTable.of(module.dialects)
    # A prepared copy's func.invoke still holds the kernel it was copied from
    # (PreparedModule.sources), so that is what each symbol stands for here.
    symbols = {module.symbol(source): source for source in module.sources}
    parsed = [op for op in operations(tree) if op.name != "builtin.module"]
    prepared = [stmt for method in module.methods for stmt in method.code.walk()]
    assert len(parsed) == len(prepared)
    for op, stmt in zip(parsed, prepared):
        assert table.statement(op.name) is type(stmt)
        entries = {k: v for k, v in op.attributes.items() if k != SEGMENT_SIZES}
        attributes = import_statement_attributes(type(stmt), entries, symbols)
        assert attributes.keys() == stmt.attributes.keys()
        for key, value in stmt.attributes.items():
            assert same(attributes[key], value), (stmt, key)
        assert list(map(import_type, op.operand_types)) == [
            arg.type for arg in stmt.args
        ]
        assert list(map(import_type, op.result_types)) == [
            result.type for result in stmt.results
        ]
        assert [
            import_type(typ)
            for region in op.regions
            for block in region.blocks
            for _, typ in block.arguments
        ] == block_argument_types(stmt)


def ours(text: str) -> str:
    return text


#: The two writers the import reads: the emitter, and ``mlir-opt``'s generic
#: printout of the emitter's text.
WRITERS = [
    pytest.param(ours, id="ours"),
    pytest.param(generic, id="mlir-opt", marks=needs_mlir),
]


@pytest.mark.parametrize("write", WRITERS)
@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_every_type_and_attribute_reads_back(kernel, write):
    module = prepare_for_mlir(kernel)
    tree = parse_module(write(emit_mlir(module)))
    assert_types_and_attributes_read_back(module, tree)


# -- step 3: each kind of type -------------------------------------------------


def test_every_kirin_type_mnemonic_has_an_importer():
    assert TYPE_IMPORTERS.keys() == TYPE_SCHEMA.keys()


def back(typ):
    """``typ`` exported and imported again."""
    return import_type(parse_type(emit_type(typ)))


TYPES = [
    types.Any,
    types.Bottom,
    types.Int,
    types.NoneType,
    Union([types.Int, types.String]),
    Union([Literal(0, types.Int), types.NoneType]),
    Literal(3, types.Int),
    Literal("s", types.String),
    Literal(1, types.Float),
    Literal((1, "a"), Generic(tuple, types.Int, types.String)),
    Generic(list, types.Int),
    Generic(tuple, types.Int, Vararg(types.String)),
    Generic(dict, types.String, types.Int),
    Generic(ilist.IList, types.Int, Literal(2, types.Int)),
    types.MethodType,
    FunctionType((types.Int, types.String), types.Bool),
    FunctionType((), Generic(tuple, types.Int, types.Int)),
]


@pytest.mark.parametrize("typ", TYPES, ids=repr)
def test_every_kind_of_kirin_type_comes_back(typ):
    result = back(typ)
    assert type(result) is type(typ)
    assert result == typ


def test_the_exports_two_losses_come_back_as_what_was_written():
    """Not ambiguities the import could resolve: the text does not hold them.

    MLIR has no type variables, so a ``TypeVar`` is written as its bound. An
    unspecified return type is written as ``!kirin.any``, which Kirin does not
    count as equal to ``None``. A prepared module has neither today.
    """
    assert back(TypeVar("T")) == types.Any
    assert back(TypeVar("N", types.Int)) == types.Int
    assert back(types.List) == Generic(list, types.Any)
    unspecified = FunctionType((types.Int,), None)
    assert back(unspecified) == FunctionType((types.Int,), types.Any)
    assert back(unspecified) != unspecified


class Sugared:
    """Registered with a display name and prefix of its own."""


SUGARED = PyClass(Sugared, display_name="Sweet", prefix="test")


class Unseen:
    class Inner:
        """Never made a Kirin type before the test reads its name."""


def test_a_class_comes_back_as_kirins_own_type_with_its_sugar():
    assert pyclass_named(qualified_name(Sugared)) is SUGARED
    # No module holds NoneType as an attribute; Kirin's table does.
    assert pyclass_named("builtins.NoneType") is types.NoneType


def test_a_class_kirin_has_not_seen_is_found_among_loaded_modules():
    # Checks the precondition through Kirin's private table, the one the
    # import searches first, so that this test exercises the second search.
    assert Unseen.Inner not in PyClass._cache
    assert pyclass_named(qualified_name(Unseen.Inner)).typ is Unseen.Inner


def test_a_class_is_never_imported_to_find_it():
    """``import this`` prints the Zen of Python; the text must not make it."""
    assert "this" not in sys.modules
    with pytest.raises(MLIRImportError, match="never imports"):
        pyclass_named("this.Zen")
    assert "this" not in sys.modules


@pytest.mark.parametrize(
    "text, reason",
    [
        ("i64", "the builtin type i64"),
        ("!stim.qubit", "!stim.qubit, which is not a Kirin type"),
        ("!kirin.qubit", "not one of the !kirin"),
        ("!kirin.any<1>", "takes 0 parameter"),
        ("!kirin.pyclass<1>", "qualified name"),
        ('!kirin.pyclass<"no.such.Class">', "no class named 'no.such.Class'"),
        ("!kirin.union<[!kirin.any]>", "at least two members"),
        ("!kirin.vararg<!kirin.any>", "only as the last argument"),
        ("!kirin.generic<!kirin.any, []>", "body of !kirin.generic"),
        (
            (
                '!kirin.generic<!kirin.pyclass<"builtins.tuple">, '
                "[!kirin.vararg<!kirin.any>, !kirin.any]>"
            ),
            "only as the last argument",
        ),
        ("!kirin.method<[], 1>", "must be a type"),
        ("!kirin.method<!kirin.any, !kirin.any>", "must be an array of types"),
        (
            (
                '!kirin.literal<[1], !kirin.generic<!kirin.pyclass<"builtins.list">, '
                '[!kirin.pyclass<"builtins.int">]>>'
            ),
            "must be hashable",
        ),
    ],
)
def test_a_type_the_exporter_never_writes_is_refused(text, reason):
    with pytest.raises(MLIRImportError, match=reason.replace("!", "\\!")):
        import_type(parse_type(text))


# -- step 3: each kind of payload ----------------------------------------------


def declared(payload) -> types.TypeAttribute:
    """The Kirin type that declares ``payload`` all the way down."""
    if isinstance(payload, tuple):
        return Generic(tuple, *map(declared, payload))
    if isinstance(payload, list):
        return Generic(list, Union(list(map(declared, payload))))
    if isinstance(payload, dict):
        return Generic(dict, types.String, Union(list(map(declared, payload.values()))))
    return PyClass(type(payload))


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=list(PAYLOADS))
def test_every_payload_comes_back_under_its_declared_type(payload):
    attr = parse_attribute(emit_python_value(payload))
    assert same(import_python_value(attr, declared(payload)), payload)


@pytest.mark.parametrize(
    "payload, hint",
    [
        ([1, 2], types.Any),
        (b"ab", types.Any),
        ((1,), Union([Generic(tuple, types.Int), Generic(list, types.Int)])),
        ([[1]], Generic(list, types.Any)),
    ],
    ids=["array", "bytes", "both declared", "inner array"],
)
def test_a_sequence_its_declaration_does_not_decide_is_refused(payload, hint):
    """Issue 06, step 0, decision 8: refused, never guessed."""
    attr = parse_attribute(emit_python_value(payload))
    with pytest.raises(MLIRImportError, match="does not say which"):
        import_python_value(attr, hint)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("1 : i32", "integer of type i32"),
        ("1.0 : f32", "float of type f32"),
        ("dense<1> : tensor<1xi16>", "tensor of i8"),
        ("!kirin.any", "as data"),
        ("[@kernel]", "inside a payload"),
        ("array<i32: 1>", "operandSegmentSizes"),
    ],
)
def test_a_payload_the_exporter_never_writes_is_refused(text, reason):
    hint = Generic(tuple, Vararg(types.Any))
    with pytest.raises(MLIRImportError, match=reason):
        import_python_value(parse_attribute(text), hint)


# -- step 3: attributes against their declarations ----------------------------

declarations = ir.Dialect("test_import")


@statement(dialect=declarations)
class Holder(ir.Statement):
    """Fields declared as attribute classes no Kirin statement declares."""

    name = "holder"
    kind: PyClass = info.attribute()
    rest: Vararg = info.attribute()


@pytest.mark.parametrize(
    "stmt, key, text, reason",
    [
        (func.Invoke, "callee", "@missing", "names no kernel"),
        (func.Invoke, "callee", "@a::@b", "nested symbol reference"),
        (func.Invoke, "nope", "true", "declares no attribute"),
        (func.Function, "signature", "1", "function type with one result"),
        (
            func.Function,
            "signature",
            "(!kirin.any) -> (!kirin.any, !kirin.any)",
            "function type with one result",
        ),
        (ilist.New, "elem_type", "1", "declares a type"),
        (Holder, "kind", "!kirin.any", "declares a PyClass, and AnyType was found"),
        (Holder, "rest", "!kirin.vararg<!kirin.any>", "no import mapping for Vararg"),
        (py.Constant, "value", "[1]", "does not say which"),
    ],
)
def test_an_attribute_the_exporter_never_writes_is_refused(stmt, key, text, reason):
    with pytest.raises(MLIRImportError, match=reason) as caught:
        import_statement_attribute(stmt, key, parse_attribute(text), {})
    assert (caught.value.stmt, caught.value.key) == (stmt, key)
    assert f"attribute {key!r}]" in str(caught.value)
    assert str(caught.value.at(7)).startswith("line 7: ")


def test_a_kernel_takes_its_declared_type_or_its_signature():
    """Issue 06, step 0, decision 7, the two cases side by side."""
    kernel = prepare_for_mlir(CORPUS[0]).entry
    symbols = {"k": kernel}
    callee = import_statement_attribute(
        func.Invoke, "callee", parse_attribute("@k"), symbols
    )
    assert callee == ir.PyAttr(kernel, PyClass(ir.Method))
    value = import_statement_attribute(
        py.Constant, "value", parse_attribute("@k"), symbols
    )
    assert value == ir.PyAttr(
        kernel, FunctionType(kernel.arg_types, kernel.return_type)
    )
