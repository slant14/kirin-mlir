"""IRDL for Kirin dialects, generated from their statement declarations.

Issue 04. ``mlir-opt --irdl-file`` loads IRDL at run time, which makes a
dialect *registered* in real MLIR without writing C++: its operations are then
verified, not just parsed. Every declaration here is generated from Kirin's
own field schema (``kirin.decl.fields``), so a dialect's MLIR description
cannot drift from its Python definition. Change a statement and the IRDL
changes with it.

What IRDL in MLIR 23 can and cannot say decides the shape of the output
(``docs/assessment.md``, 2026-10-06 addendum; ADR 0002):

**Leaf dialects only.** IRDL has no traits and no successors, so no IRDL op
can be a terminator, branch, or hold a usable region. A registered dialect
refuses any op it does not declare, so registration is all or nothing. A
dialect is registered only if none of its statements has a region,
successors or ``IsTerminator``. Every other dialect is left out and reported,
along with the statement that excluded it, and its ops are parsed under
``--allow-unregistered-dialect`` and not verified.

**Counts, presence and well-formed types, not type constraints.** Each
operand, result and attribute is constrained by ``irdl.any``, with its own
variable. What gets verified is the number of operands and results, that
every declared attribute is present, and, through the ``@kirin`` type dialect,
that every ``!kirin.*`` type is well formed. Kirin's field types are lattice
bounds (an operand typed ``bool`` may be ``Literal(True)`` or a ``Union`` of
literals), and IRDL cannot check subtyping. A ``TypeVar`` shared between
fields must not become a shared IRDL variable, because IRDL reads that as
"equal" and Kirin does not: ``x + 1.5`` is ``py.binop.add(int, float) ->
float``, though ``lhs``, ``rhs`` and ``result`` all declare ``T``.

**Shapes the fields do not declare.** Kirin's generated ``__init__`` creates
exactly the declared fields. A hand-written one can create more:
``py.tuple.new`` declares no argument fields but takes its elements as
operands, and ``py.unpack.unpack`` declares no result fields but has one result
per name. Where a hand-written ``__init__`` declares no operands (or no
results), one variadic group is declared instead. That checks less, but it
never rejects what Kirin can build. Each case is reported.

**Every trait is reported as not expressed**, because IRDL expresses none:
``Pure``, ``IsTerminator``, ``SSACFG`` and the rest. Issue 07 phase A takes
the report as its input.

The report goes at the head of the generated file, as comments, so the file
says what it does not check. It is also returned as :class:`GeneratedIRDL` for
code to read.
"""

from __future__ import annotations

from importlib import metadata
from dataclasses import field, dataclass
from collections.abc import Iterable

from kirin import ir
from kirin.decl import fields

from .names import NameTable, checked
from .types import DIALECT, TYPE_SCHEMA
from .errors import MLIRExportError


@dataclass(frozen=True)
class Exclusion:
    """Why a dialect cannot be registered: one statement, and what it has."""

    dialect: str
    statement: str
    reason: str

    def __str__(self) -> str:
        return f"{self.statement} {self.reason}"


@dataclass(frozen=True)
class GeneratedIRDL:
    """IRDL text, with what it registered and what it could not express.

    Args:
        text: the file to hand to ``mlir-opt --irdl-file``.
        registered: Kirin dialects declared in it, by Kirin name.
        excluded: dialects left out, each with the statement responsible.
        traits: statement to the traits of it that IRDL cannot express.
        loose: statement to ``"operands"`` and/or ``"results"``, declared as
            one variadic group because its fields do not declare them.
    """

    text: str
    registered: tuple[str, ...]
    excluded: tuple[Exclusion, ...]
    traits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    loose: dict[str, tuple[str, ...]] = field(default_factory=dict)


def exclusion(dialect: ir.Dialect) -> Exclusion | None:
    """The first statement that keeps ``dialect`` out of IRDL, if any.

    ADR 0002 decision 2: registered if and only if no statement has a region,
    successors, or the ``IsTerminator`` trait.
    """
    for stmt in dialect.stmts:
        schema = fields(stmt)
        if schema.regions:
            reason = "has a region"
        elif schema.blocks:
            reason = "has successors"
        elif stmt.has_trait(ir.IsTerminator):
            reason = "is a terminator"
        else:
            continue
        return Exclusion(dialect.name, f"{dialect.name}.{stmt.name}", reason)
    return None


def generate_irdl(dialects: ir.DialectGroup | Iterable[ir.Dialect]) -> GeneratedIRDL:
    """IRDL for every leaf dialect among ``dialects``, plus the ``!kirin.*`` types.

    Dialects that cannot be registered are left out and reported, never
    approximated.

    Raises:
        MLIRExportError: if a name (dialect, statement or field) is not one
            IRDL accepts, or the type schema has a parameter kind this module
            was not taught.
    """
    if isinstance(dialects, ir.DialectGroup):
        dialects = dialects.data
    dialects = sorted(set(dialects), key=lambda d: d.name)
    names = NameTable.of(dialects)

    registered: list[ir.Dialect] = []
    excluded: list[Exclusion] = []
    for dialect in dialects:
        reason = exclusion(dialect)
        if reason is None:
            registered.append(dialect)
        else:
            excluded.append(reason)

    traits: dict[str, tuple[str, ...]] = {}
    loose: dict[str, tuple[str, ...]] = {}
    body = [_type_dialect()]
    for dialect in registered:
        ops = []
        for stmt in dialect.stmts:
            full = f"{dialect.name}.{stmt.name}"
            text, missing = _operation(names.operation(stmt).split(".", 1)[1], stmt)
            ops.append(text)
            if stmt.traits:
                traits[full] = tuple(sorted(type(t).__name__ for t in stmt.traits))
            if missing:
                loose[full] = missing
        body.append(f"irdl.dialect @{names.namespace(dialect)} {{\n{''.join(ops)}}}\n")

    header = _report(excluded, traits, loose)
    return GeneratedIRDL(
        header + "\n" + "".join(body),
        tuple(d.name for d in registered),
        tuple(excluded),
        traits,
        loose,
    )


def _operation(op: str, stmt: type[ir.Statement]) -> tuple[str, tuple[str, ...]]:
    """One ``irdl.operation``, and the parts its fields did not declare."""
    schema = fields(stmt)
    # Kirin compiles a generated __init__ from a string; anything else was
    # written by hand, and may build operands or results its fields omit.
    code = getattr(stmt.__init__, "__code__", None)
    handwritten = getattr(code, "co_filename", None) != "<string>"
    lines: list[str] = []
    missing: list[str] = []

    def variable() -> str:
        # A fresh variable per field: reusing one would mean "equal".
        name = f"%{len(lines)}"
        lines.append(f"    {name} = irdl.any\n")
        return name

    what = f"statement {stmt.dialect.name if stmt.dialect else '?'}.{stmt.name}"
    operands = [
        f"{checked(arg.name, f'operand {arg.name!r} of {what}')}: "
        f"{'variadic ' if arg.group else ''}{variable()}"
        for arg in schema.args.values()
    ]
    if not operands and handwritten:
        operands = [f"args: variadic {variable()}"]
        missing.append("operands")

    results = [
        f"{checked(name, f'result {name!r} of {what}')}: {variable()}"
        for name in schema.results
    ]
    if not results and handwritten:
        results = [f"results: variadic {variable()}"]
        missing.append("results")

    attributes = [f'"{key}" = {variable()}' for key in schema.attributes]

    if operands:
        lines.append(f"    irdl.operands({', '.join(operands)})\n")
    if results:
        lines.append(f"    irdl.results({', '.join(results)})\n")
    if attributes:
        lines.append(f"    irdl.attributes {{{', '.join(attributes)}}}\n")
    return f"  irdl.operation @{op} {{\n{''.join(lines)}  }}\n", tuple(missing)


def _type_dialect() -> str:
    """``irdl.dialect @kirin``: the nine ``!kirin.*`` types of ``TYPE_SCHEMA``."""
    types = []
    for schema in TYPE_SCHEMA.values():
        if not schema.parameters:
            types.append(f"  irdl.type @{schema.mnemonic}\n")
            continue
        lines, params = [], []
        for index, description in enumerate(schema.parameters):
            name, constraint = _parameter(schema.mnemonic, description)
            lines.append(f"    %{index} = {constraint}\n")
            params.append(f"{name}: %{index}")
        lines.append(f"    irdl.parameters({', '.join(params)})\n")
        types.append(f"  irdl.type @{schema.mnemonic} {{\n{''.join(lines)}  }}\n")
    return f"irdl.dialect @{DIALECT} {{\n{''.join(types)}}}\n"


def _parameter(mnemonic: str, description: str) -> tuple[str, str]:
    """Read ``"name: Kind, prose"`` from ``TYPE_SCHEMA`` as an IRDL constraint."""
    name, _, rest = description.partition(":")
    kind = rest.split(",")[0].strip()
    if kind == "StringAttr":
        constraint = 'irdl.base "#builtin.string"'
    elif kind.startswith("ArrayAttr"):
        constraint = 'irdl.base "#builtin.array"'
    elif kind in ("AnyAttr", "AnyType"):
        constraint = "irdl.any"
    elif kind.startswith(f"!{DIALECT}.") and kind[len(DIALECT) + 2 :] in TYPE_SCHEMA:
        constraint = f"irdl.base @{DIALECT}::@{kind[len(DIALECT) + 2:]}"
    else:
        raise MLIRExportError(
            f"!{DIALECT}.{mnemonic} has a parameter of kind {kind!r}, which "
            "kirin_mlir.irdl does not know how to constrain"
        )
    return checked(name.strip(), f"parameter of !{DIALECT}.{mnemonic}"), constraint


def _report(
    excluded: list[Exclusion],
    traits: dict[str, tuple[str, ...]],
    loose: dict[str, tuple[str, ...]],
) -> str:
    lines = [
        "IRDL for Kirin dialects, generated by kirin-mlir from the statement",
        f"declarations of kirin-toolchain {_kirin_version()}. Do not edit; regenerate.",
        "",
        "Checked: operand and result counts, that each declared attribute is",
        "present, and that every !kirin.* type is well formed.",
        "",
        "Not checked: operand, result and attribute types. Each is irdl.any, with",
        "its own variable. Kirin's field types are lattice bounds, which IRDL",
        "cannot check, and a TypeVar shared between fields does not mean equal",
        "types in Kirin.",
    ]
    if excluded:
        lines += [
            "",
            "Not registered: IRDL cannot declare a region, successors or a",
            "terminator (ADR 0002, decision 2). These dialects' ops need",
            "--allow-unregistered-dialect and are parsed, not verified.",
        ]
        width = max(len(e.dialect) for e in excluded)
        lines += [f"  {e.dialect:<{width}}  {e}" for e in excluded]
    if loose:
        lines += [
            "",
            "Not declared by the statement's fields, so declared as one variadic",
            "group (a hand-written __init__ can create any number):",
        ]
        width = max(map(len, loose))
        lines += [f"  {s:<{width}}  {', '.join(parts)}" for s, parts in loose.items()]
    if traits:
        lines += ["", "Traits not expressed (IRDL has none; issue 07 phase A):"]
        width = max(map(len, traits))
        lines += [f"  {s:<{width}}  {', '.join(names)}" for s, names in traits.items()]
    return "".join(f"// {line}".rstrip() + "\n" for line in lines)


def _kirin_version() -> str:
    try:
        return metadata.version("kirin-toolchain")
    except metadata.PackageNotFoundError:
        return "(unknown version)"
