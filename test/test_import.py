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

Step 4 rebuilds statements, blocks and regions. Each kernel of the corpus,
again from both writers, must come back structurally equal to its prepared
copy by Kirin's own ``is_structurally_equal``, with what that does not
compare checked as well: the operands each argument field reads, result
types, the statement's own state, every reverse-use link (Kirin issue #707),
and Kirin's ``verify``. Small hand-written modules check forward references,
scoping, successors and the refusals.
"""

import sys
import struct
from collections import Counter, defaultdict

import pytest
from kirin import ir, types
from oracle import MLIR_OPT, SKIP_REASON
from test_irdl import CORPUS
from kirin.decl import info, statement
from test_parse import PAYLOADS, generic, operations
from kirin.prelude import basic
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
    MLIRImporter,
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
from kirin_mlir.import_ import STATEMENT_IMPORTS

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


# -- step 4: statements, blocks and regions, over the corpus -----------------


def rebuilt(module, text: str) -> list[ir.Statement]:
    """The statements ``text`` holds, read back for ``module``'s dialects."""
    symbols = {module.symbol(source): source for source in module.sources}
    importer = MLIRImporter(NameTable.of(module.dialects), symbols)
    return importer.module(parse_module(text))


def operand_groups(stmt: ir.Statement) -> dict[str, int | tuple[int, ...]]:
    """The operand positions each argument field reads.

    Not ``_name_args_slice`` itself: Kirin's hand-written ``scf`` constructors
    record a trailing group as ``slice(1, None)`` and generated ones as
    ``slice(1, 3)``. Both read the same operands, and the text cannot say
    which constructor built the original.
    """
    positions = tuple(range(len(stmt.args)))
    return {name: positions[index] for name, index in stmt._name_args_slice.items()}


def assert_use_links(stmt: ir.Statement) -> None:
    """Each value's reverse uses are exactly the operands that read it."""
    readers = defaultdict(set)
    defined = set()
    for node in stmt.walk():
        defined.update(node.results)
        for region in node.regions:
            for block in region.blocks:
                defined.update(block.args)
        for index, arg in enumerate(node.args):
            assert not isinstance(arg, ir.TestValue), "a placeholder was left"
            readers[arg].add((node, index))
    assert readers.keys() <= defined, "an operand is defined outside the kernel"
    for value in defined:
        assert {(use.stmt, use.index) for use in value.uses} == readers[value]


def assert_rebuilt(module, statements: list[ir.Statement]) -> None:
    """``statements`` are ``module``'s prepared kernels, rebuilt."""
    assert len(statements) == len(module.methods)
    for stmt, method in zip(statements, module.methods):
        assert stmt.is_structurally_equal(method.code)
        assert_use_links(stmt)
        self_arg, prepared_self = (
            code.body.blocks[0].args[0] for code in (stmt, method.code)
        )
        assert self_arg.name == prepared_self.name
        for imported, prepared in zip(stmt.walk(), method.code.walk(), strict=True):
            assert operand_groups(imported) == operand_groups(prepared)
            assert [r.type for r in imported.results] == [
                r.type for r in prepared.results
            ]
            assert vars(imported).keys() == vars(prepared).keys()
            imported.verify()
            imported.verify_type()


@pytest.mark.parametrize("write", WRITERS)
@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_every_kernel_is_rebuilt_as_its_prepared_copy(kernel, write):
    module = prepare_for_mlir(kernel)
    assert_rebuilt(module, rebuilt(module, write(emit_mlir(module))))


def test_func_func_is_the_one_statement_with_an_import_hook():
    """The import's one exception, as ``func.func`` is the emitter's."""
    assert STATEMENT_IMPORTS.keys() == {func.Function}


# -- step 4: hand-written modules ---------------------------------------------


@statement(dialect=declarations)
class Scope(ir.Statement):
    """A region and nothing else, for the scoping tests."""

    name = "scope"
    body: ir.Region = info.region()


@statement(dialect=declarations)
class Groups(ir.Statement):
    """One operand, then two groups, so it carries ``operandSegmentSizes``."""

    name = "groups"
    first: ir.SSAValue = info.argument()
    xs: tuple[ir.SSAValue, ...] = info.argument()
    ys: tuple[ir.SSAValue, ...] = info.argument()


TABLE = NameTable.of([*basic.data, declarations])
INT = '!kirin.pyclass<"builtins.int">'
CONSTANT = f'%c = "kirin_py_constant.constant"() {{value = 1 : i64}} : () -> {INT}'


def add(lhs: str, rhs: str, typ: str = INT) -> str:
    return f'%s = "kirin_py_binop.add"({lhs}, {rhs}) : ({typ}, {typ}) -> {INT}'


def scope(*lines: str) -> list[str]:
    return ['"kirin_test_import.scope"() ({', *lines, "}) : () -> ()"]


def groups(operands: str, sizes: str | None) -> str:
    """A ``Groups`` statement; ``sizes`` is its ``operandSegmentSizes``, if any."""
    types_ = ", ".join(INT for _ in operands.split(", "))
    entries = f" {{operandSegmentSizes = {sizes}}}" if sizes else ""
    return f'"kirin_test_import.groups"({operands}){entries} : ({types_}) -> ()'


def module_text(*lines: str) -> str:
    """A ``builtin.module`` holding ``lines``, the first of them on line 2."""
    body = "".join(f"  {line}\n" for line in lines)
    return f'"builtin.module"() ({{\n{body}}}) : () -> ()\n'


def rebuild(*lines: str) -> list[ir.Statement]:
    return MLIRImporter(TABLE).module(parse_module(module_text(*lines)))


def test_a_value_may_be_used_before_its_definition():
    plus, one = rebuild(add("%c", "%c"), CONSTANT)
    assert tuple(plus.args) == (one.result, one.result)
    assert {(use.stmt, use.index) for use in one.result.uses} == {
        (plus, 0),
        (plus, 1),
    }


@pytest.mark.parametrize("before", [True, False], ids=["defined first", "used first"])
def test_a_nested_region_sees_the_values_around_it(before):
    inner = scope(add("%c", "%c"))
    lines = [CONSTANT, *inner] if before else [*inner, CONSTANT]
    statements = rebuild(*lines)
    one = statements[0] if before else statements[1]
    region = statements[1] if before else statements[0]
    (plus,) = region.body.blocks[0].stmts
    assert tuple(plus.args) == (one.result, one.result)


def test_a_successor_may_name_a_later_block_of_its_region():
    (outer,) = rebuild(
        *scope(
            "^bb0:",
            '  "kirin_cf.br"() [^bb1] : () -> ()',
            "^bb1:",
            '  "kirin_cf.br"() [^bb0] : () -> ()',
        )
    )
    first, second = outer.body.blocks
    assert first.last_stmt.successors == [second]
    assert second.last_stmt.successors == [first]


def test_operand_groups_follow_operand_segment_sizes():
    (stmt,) = rebuild(*scope(CONSTANT, groups("%c, %c, %c", "array<i32: 1, 0, 2>")))
    _, rebuilt_groups = stmt.body.blocks[0].stmts
    assert rebuilt_groups._name_args_slice == {
        "first": 0,
        "xs": slice(1, 1),
        "ys": slice(1, 3),
    }
    assert rebuilt_groups.xs == () and len(rebuilt_groups.ys) == 2
    assert SEGMENT_SIZES not in rebuilt_groups.attributes


INT3 = f"({INT}, {INT}, {INT})"
#: Each case: the module's lines, what the refusal says, and the line it names.
REFUSALS = {
    "use without definition": ([add("%c", "%c")], "%c is used but never defined", 2),
    "definition twice": ([CONSTANT, CONSTANT], "defined twice", 3),
    "use of another type": (
        [CONSTANT, add("%c", "%c", "!kirin.any")],
        r"used as AnyType\(\) and defined as",
        3,
    ),
    "use before definition of another type": (
        [add("%c", "%c", "!kirin.any"), CONSTANT],
        r"used as AnyType\(\) and defined as",
        3,
    ),
    "two uses of different types": (
        [add("%c", "%c", "!kirin.any"), add("%c", "%c").replace("%s", "%t", 1)],
        "used before as",
        3,
    ),
    "use outside its region": (
        [*scope(CONSTANT), add("%c", "%c")],
        "never defined",
        5,
    ),
    "definition in a nested region": (
        [add("%c", "%c"), *scope(CONSTANT)],
        "outside the region that defines",
        4,
    ),
    "no such block": (
        scope("^bb0:", '  "kirin_cf.br"() [^bb7] : () -> ()'),
        "names no block",
        4,
    ),
    "a block of another region": (
        [*scope("^bb0:", f"  {CONSTANT}"), *scope('"kirin_cf.br"() [^bb0] : () -> ()')],
        "names no block",
        7,
    ),
    "a block of an enclosing region": (
        scope(
            "^bb0:",
            *(f"  {line}" for line in scope('"kirin_cf.br"() [^bb0] : () -> ()')),
        ),
        "names no block",
        5,
    ),
    "a label twice": (scope("^bb0:", "^bb0:"), "defined twice in one region", 4),
    "segment sizes missing": ([CONSTANT, groups("%c", None)], "is missing", 3),
    "segment sizes not needed": (
        [
            CONSTANT,
            (
                f'%s = "kirin_py_binop.add"(%c, %c) '
                f"{{operandSegmentSizes = array<i32: 1, 1>}} : ({INT}, {INT}) -> {INT}"
            ),
        ],
        "written only for",
        3,
    ),
    "segment sizes of i64": (
        [CONSTANT, groups("%c", "array<i64: 1, 0, 0>")],
        "must be an array<i32",
        3,
    ),
    "segment sizes too few": (
        [CONSTANT, groups("%c", "array<i32: 1, 0>")],
        "has 2 entries",
        3,
    ),
    "segment sizes over a single operand": (
        [CONSTANT, groups("%c, %c", "array<i32: 2, 0, 0>")],
        "gives the field 'first' 2",
        3,
    ),
    "segment sizes over the operands": (
        [CONSTANT, groups("%c, %c", "array<i32: 1, 1, 1>")],
        "take 3 operands, and the operation has 2",
        3,
    ),
    "operands over the fields": (
        [CONSTANT, f'%s = "kirin_py_binop.add"(%c, %c, %c) : {INT3} -> {INT}'],
        "take 2 operands, and the operation has 3",
        3,
    ),
    "results under the fields": (
        [CONSTANT, f'"kirin_py_binop.add"(%c, %c) : ({INT}, {INT}) -> ()'],
        "declares 1 result",
        3,
    ),
    "a property": (
        [f'%c = "kirin_py_constant.constant"() <{{value = 1 : i64}}> : () -> {INT}'],
        "property",
        2,
    ),
    "a func.func with no block": (
        [
            (
                '%f = "kirin_func.func"() ({\n  }) {sym_name = "f", slots = [], '
                "signature = () -> (!kirin.any)} : () -> !kirin.anymethod"
            )
        ],
        "no entry block",
        2,
    ),
    "an unknown name": (
        ['"kirin_py_binop.nope"() : () -> ()'],
        "names no statement",
        2,
    ),
}


@pytest.mark.parametrize("lines, reason, line", REFUSALS.values(), ids=list(REFUSALS))
def test_ir_the_exporter_never_writes_is_refused_with_its_line(lines, reason, line):
    with pytest.raises(MLIRImportError, match=reason) as caught:
        rebuild(*lines)
    assert caught.value.line == line


def test_a_module_is_one_region_of_one_block():
    text = '"builtin.module"() ({\n^bb0(%a: !kirin.any):\n}) : () -> ()\n'
    with pytest.raises(MLIRImportError, match="one region of one block"):
        MLIRImporter(TABLE).module(parse_module(text))
