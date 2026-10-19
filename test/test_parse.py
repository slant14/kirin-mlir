"""Reading MLIR generic form back into a tree: issue 06, step 1.

The parser has two writers to read: this package's emitter, and ``mlir-opt
--mlir-print-op-generic``, which spells the same IR differently (see
:mod:`kirin_mlir.parse`). The strongest checks here parse both and require the
same tree once every value and block is renamed in the order it is defined.
That compares two parses, not text with expected text, and one of the two is
of what real MLIR wrote.
"""

import re
import math
from collections import Counter
from dataclasses import replace

import pytest
from kirin import types
from oracle import (
    MLIR_OPT,
    SKIP_REASON,
    mlir_opt,
    module_with_types,
    module_with_attributes,
)
from test_irdl import CORPUS
from kirin.ir.attrs.types import Union, Vararg, Generic, Literal

from kirin_mlir import (
    NameTable,
    MLIRImportError,
    emit_mlir,
    emit_type,
    parse_type,
    parse_module,
    escape_string,
    parse_attribute,
    prepare_for_mlir,
    emit_python_value,
)
from kirin_mlir.parse import (
    BoolAttr,
    TypeAttr,
    UnitAttr,
    ArrayAttr,
    FloatAttr,
    FloatType,
    Operation,
    StringAttr,
    TensorType,
    DialectType,
    IntegerAttr,
    IntegerType,
    FunctionType,
    SymbolRefAttr,
    DenseArrayAttr,
    DictionaryAttr,
    DenseElementsAttr,
)

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)


def ids(kernel) -> str:
    return kernel.sym_name


def generic(text: str, *flags: str) -> str:
    """``text`` as ``mlir-opt`` prints it back in generic form."""
    result = mlir_opt(
        text, "--allow-unregistered-dialect", "--mlir-print-op-generic", *flags
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def operations(op: Operation):
    """``op`` and every operation nested in it, outermost first."""
    yield op
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield from operations(inner)


def canonical(module: Operation) -> Operation:
    """``module`` with its values and blocks renamed in the order defined.

    ``mlir-opt`` renumbers both, so two spellings of one module compare equal
    only after this. Values are named across the whole module, and blocks per
    region, which is how MLIR scopes them here. An unlabelled entry block gets
    a label too, since ``mlir-opt`` leaves out the entry label when the block
    has no arguments.
    """
    values: dict[str, str] = {}
    for op in operations(module):
        for name in op.results:
            values[name] = f"%v{len(values)}"
        for region in op.regions:
            for block in region.blocks:
                for name, _ in block.arguments:
                    values[name] = f"%v{len(values)}"

    def rename(op: Operation, labels: dict) -> Operation:
        return replace(
            op,
            results=tuple(values[name] for name in op.results),
            operands=tuple(values.get(name, name) for name in op.operands),
            successors=tuple(labels[label] for label in op.successors),
            regions=tuple(rename_region(region) for region in op.regions),
        )

    def rename_region(region):
        labels = {block.label: f"^b{i}" for i, block in enumerate(region.blocks)}
        blocks = tuple(
            replace(
                block,
                label=labels[block.label],
                arguments=tuple((values[n], t) for n, t in block.arguments),
                operations=tuple(rename(op, labels) for op in block.operations),
            )
            for block in region.blocks
        )
        return replace(region, blocks=blocks)

    return rename(module, {})


def wrap(body: str) -> str:
    return f'"builtin.module"() ({{\n  {body}\n}}) : () -> ()\n'


def only_operation(module: Operation) -> Operation:
    return module.regions[0].blocks[0].operations[0]


# -- the emitter's modules ----------------------------------------------------


@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_every_emitted_operation_is_read_back_under_its_name(kernel):
    """One operation in the tree per Kirin statement, so nothing is skipped."""
    module = prepare_for_mlir(kernel)
    parsed = Counter(op.name for op in operations(parse_module(emit_mlir(module))))
    assert parsed.pop("builtin.module") == 1

    names = NameTable.of(module.dialects)
    expected = Counter(
        names.operation(stmt)
        for method in module.methods
        for stmt in method.code.walk()
    )
    assert parsed == expected


@needs_mlir
@pytest.mark.parametrize("kernel", CORPUS, ids=ids)
def test_mlir_opts_printout_reads_as_the_same_tree(kernel):
    text = emit_mlir(prepare_for_mlir(kernel))
    assert canonical(parse_module(generic(text))) == canonical(parse_module(text))


# -- attributes and types -----------------------------------------------------

#: Every kind of payload the exporter writes, and each spelling ``mlir-opt``
#: changes: escapes, splat and hex ``dense``, bare unit keys, untyped numbers
#: in arrays, float formatting, NaN and infinities as bit patterns.
PAYLOADS = {
    "none": None,
    "true": True,
    "false": False,
    "zero": 0,
    "negative": -5,
    "i64 max": 2**63 - 1,
    "i64 min": -(2**63),
    "tenth": 0.1,
    "huge": 1e300,
    "negative zero": -0.0,
    "nan": math.nan,
    "infinity": math.inf,
    "-infinity": -math.inf,
    "subnormal": 5e-324,
    "plain": "plain",
    "escapes": 'q"b\\n\nt\t\x00\x7f',
    "unicode": "héllo ✓",
    "empty string": "",
    "bytes": b"\x00\x01\xff",
    "no bytes": b"",
    "splat bytes": b"\x07\x07\x07",
    "bytearray": bytearray(b"ab"),
    "200 bytes": bytes(range(200)),
    "tuple": (1, "a", 2.5),
    "empty list": [],
    "nested": [[1], [2, [3]]],
    "dict": {"k": 1, "weird key": "v", "u": None},
    "class": int,
}


@needs_mlir
@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=list(PAYLOADS))
def test_every_payload_reads_back_the_same_from_both_writers(payload):
    text = module_with_attributes({"v": emit_python_value(payload)})
    ours = only_operation(parse_module(text)).attributes
    theirs = only_operation(parse_module(generic(text))).attributes
    assert theirs == ours


I8 = IntegerType(8)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("unit", UnitAttr()),
        ("true", BoolAttr(True)),
        ("-5 : i64", IntegerAttr(-5)),
        ("7", IntegerAttr(7, IntegerType(64))),
        ("1.000000e-01 : f64", FloatAttr(0.1)),
        ("2.5", FloatAttr(2.5, FloatType("f64"))),
        ("0x7FF0000000000000 : f64", FloatAttr(math.inf)),
        ('"a\\22b\\5Cc\\0A"', StringAttr('a"b\\c\n')),
        ("dense<> : tensor<0xi8>", DenseElementsAttr((), TensorType((0,), I8))),
        ("dense<7> : tensor<3xi8>", DenseElementsAttr((7, 7, 7), TensorType((3,), I8))),
        (
            'dense<"0x00FF7F80"> : tensor<4xi8>',
            DenseElementsAttr((0, -1, 127, -128), TensorType((4,), I8)),
        ),
        ("array<i32: 1, 1, 0>", DenseArrayAttr(IntegerType(32), (1, 1, 0))),
        ("array<i32>", DenseArrayAttr(IntegerType(32), ())),
        ("[1, [2]]", ArrayAttr((IntegerAttr(1), ArrayAttr((IntegerAttr(2),))))),
        (
            '{b, "c d" = 1 : i64}',
            DictionaryAttr({"b": UnitAttr(), "c d": IntegerAttr(1)}),
        ),
        ("@callee", SymbolRefAttr("callee")),
        ('@"odd name"::@inner', SymbolRefAttr("odd name", ("inner",))),
        ("() -> ()", TypeAttr(FunctionType((), ()))),
        ("i1", TypeAttr(IntegerType(1))),
    ],
)
def test_an_attribute_means_what_mlir_reads_it_as(text, expected):
    assert parse_attribute(text) == expected


def test_floats_compare_by_their_bits():
    """Two spellings of one float are equal, and two floats are not."""
    assert parse_attribute("-0.000000e+00 : f64") == FloatAttr(-0.0)
    assert FloatAttr(-0.0) != FloatAttr(0.0)
    assert FloatAttr(math.nan) == FloatAttr(math.nan)


@pytest.mark.parametrize(
    "text", ["", "plain", 'q"b\\n', "\x00\x1f\x7f", "héllo ✓", "tab\tnew\nline"]
)
def test_the_emitters_strings_read_back(text):
    assert parse_attribute(f'"{escape_string(text)}"') == StringAttr(text)


KIRIN_TYPES = [
    types.Any,
    types.Bottom,
    types.Int,
    Union([types.Int, types.String]),
    Literal(3),
    Literal("s", types.String),
    types.List,
    types.Tuple,
    Vararg(types.Int),
    Generic(dict, types.String, types.Int),
    types.MethodType,
    types.MethodType[[types.Int, types.String], types.Bool],
]


@needs_mlir
def test_every_kind_of_kirin_type_reads_back_the_same_from_both_writers():
    text = module_with_types([emit_type(t) for t in KIRIN_TYPES])
    ours = only_operation(parse_module(text)).result_types
    theirs = only_operation(parse_module(generic(text))).result_types
    assert theirs == ours
    assert len(ours) == len(KIRIN_TYPES)


def test_a_kirin_type_keeps_its_parameters():
    int_type = DialectType("kirin", "pyclass", (StringAttr("builtins.int"),))
    assert parse_type(emit_type(Literal(3))) == DialectType(
        "kirin", "literal", (IntegerAttr(3), TypeAttr(int_type))
    )


# -- structure ----------------------------------------------------------------


def test_a_result_group_is_named_one_value_per_result():
    module = parse_module(
        wrap('%0:2 = "t.two"() : () -> (i64, i64)\n  "t.use"(%0#1) : (i64) -> ()')
    )
    two, use = module.regions[0].blocks[0].operations
    assert two.results == ("%0#0", "%0#1")
    assert use.operands == ("%0#1",)


def test_blocks_successors_and_properties():
    text = """"builtin.module"() ({
  "t.region"() ({
  ^bb0(%a: i64):
    "t.br"(%a)[^bb1] <{p = 1 : i64}> : (i64) -> ()
  ^bb1:  // pred: ^bb0
    "t.end"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    region = only_operation(parse_module(text)).regions[0]
    entry, exit_ = region.blocks
    assert entry.label == "^bb0"
    assert entry.arguments == (("%a", IntegerType(64)),)
    branch = entry.operations[0]
    assert branch.successors == ("^bb1",)
    assert branch.properties == {"p": IntegerAttr(1)}
    assert (exit_.label, exit_.line) == ("^bb1", 5)


def test_an_entry_block_may_have_no_label():
    region = only_operation(
        parse_module(wrap('"t.r"() ({\n    "t.x"() : () -> ()\n  }) : () -> ()'))
    ).regions[0]
    assert region.blocks[0].label is None
    assert region.blocks[0].operations[0].name == "t.x"


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, message",
    [
        ("module {\n}", "custom assembly is not read"),
        (
            '#map = affine_map<(d0) -> (d0)>\n"builtin.module"() ({}) : () -> ()',
            "alias definition is not read",
        ),
        (wrap('"t.a"() {m = #map} : () -> ()'), "attribute alias or a dialect"),
        (wrap('"t.a"() : () -> () loc("f.py":1:2)'), "locations are not read"),
        (wrap('"t.a"() {m = affine_map<(d0) -> (d0)>} : () -> ()'), "affine_map"),
        (wrap('"t.a"() : () -> memref<4xi8>'), "memref is not one the exporter"),
        (wrap('"t.a"() : () -> !alias'), "type alias or an opaque dialect type"),
        (wrap('"t.a"() {d = dense<1.5> : tensor<1xf32>} : () -> ()'), "integer dense"),
        (wrap('"t.a"() {s = "\\FF"} : () -> ()'), "not UTF-8"),
        (wrap('"t.a"() {s = "\\q"} : () -> ()'), "unknown escape"),
        (wrap('"t.a"() {s = "open} : () -> ()'), "not closed"),
        (wrap('"t.a"() {k = 1, k = 2} : () -> ()'), "appears twice"),
        (wrap('%0 = "t.a"() : () -> ()'), "1 results but 0 result types"),
        (wrap('"t.a"(%0) : () -> ()'), "1 operands but 0 operand types"),
        (wrap('"t.a"()[^bb1(%0 : i64)] : () -> ()'), "successor operands"),
        ('"t.a"() : () -> ()', 'expected one "builtin.module"'),
        ('"builtin.module"() ({\n  "t.a"() : () -> ()\n', "region is not closed"),
        (
            '"builtin.module"() ({}) : () -> ()\n"t.b"() : () -> ()',
            "end of the input after the module",
        ),
    ],
)
def test_what_the_exporter_never_writes_is_refused(text, message):
    with pytest.raises(MLIRImportError, match=re.escape(message)):
        parse_module(text)


def test_a_refusal_says_where():
    with pytest.raises(MLIRImportError) as caught:
        parse_module(wrap('"t.a"() : () -> memref<4xi8>'))
    assert (caught.value.line, caught.value.column) == (2, 19)
    assert str(caught.value).startswith("line 2, column 19: ")
