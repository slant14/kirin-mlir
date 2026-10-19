"""MLIR generic form back into Kirin IR: the second stage of the import.

Issue 06. :mod:`kirin_mlir.parse` turns text into a tree that knows nothing
about Kirin; this module turns the tree into Kirin statements, blocks and
regions. It mirrors the emitter's one generic path: generic form spells every
operation the same way, so every statement is rebuilt the same way, through
Kirin's base ``Statement.__init__``, the way Kirin's own
``Statement.from_stmt`` builds one (issue 06, step 0, decision 4). A
statement's hand-written ``__init__`` is never called: it takes the fields of
one particular statement, and the text has already decided all of them.

What the generic path decides, beyond the base constructor:

**Names.** An operation's name comes back as its statement class through the
export's own :class:`NameTable` (step 2), and its types and attributes through
the inverses beside their emitters (step 3).

**Argument groups.** Kirin records which operands belong to which argument
field in ``_name_args_slice``, and a statement's named properties read their
operands through it, so it has to be rebuilt exactly. With no variadic field,
each field takes one operand; with one, that field takes what the others
leave; with two or more, ``operandSegmentSizes`` says, and it is then dropped:
it describes the operands, and is not an attribute of the statement.

**Values used before their definition.** MLIR allows a use to come first in
the text, so such a use gets a placeholder, which the definition replaces
through Kirin's ``SSAValue.replace_by``. That keeps every reverse-use link
right, which Kirin issue #707 found its own JSON deserializer did not. A value
is visible in the region that defines it and the regions nested in it, as
MLIR's parser scopes names. A use with no definition in scope, a second
definition, and a use whose type differs from the definition's are refused.

**Blocks.** A label names a block of the region it is in, as ``mlir-opt``
numbers them. Every block of a region exists before any of its operations is
read, so a successor may name a block that comes later.

**One exception.** A statement class may register a hook through
:func:`register_statement_import` for what the generic path cannot rebuild.
Kirin ``func.func`` is the one: its ``%self`` argument was dropped on the way
out, and :mod:`kirin_mlir.dialects.func` puts it back.
"""

from __future__ import annotations

from dataclasses import field, dataclass
from collections.abc import Mapping, Callable

from kirin import ir
from kirin.decl import fields

from . import parse
from .emit import SEGMENT_SIZES
from .attrs import import_statement_attributes
from .names import NameTable
from .types import import_type
from .errors import MLIRImportError

#: Hooks run on a statement once the generic path has built it, keyed by
#: statement class. Populated by :func:`register_statement_import`.
STATEMENT_IMPORTS: dict[type[ir.Statement], Callable[[ir.Statement], None]] = {}


def register_statement_import(
    cls: type[ir.Statement],
) -> Callable[[Callable[[ir.Statement], None]], Callable[[ir.Statement], None]]:
    """Register what to finish on a ``cls`` after the generic path built it.

    The import's counterpart of a per-dialect ``emit.mlir`` method table: only
    for what generic form does not carry. Looked up by exact class, as the
    emitter's name table is.
    """

    def wrapper(fn: Callable[[ir.Statement], None]) -> Callable[[ir.Statement], None]:
        STATEMENT_IMPORTS[cls] = fn
        return fn

    return wrapper


@dataclass
class _Forward:
    """A value used before its definition."""

    placeholder: ir.TestValue
    line: int
    #: The regions open at every use so far. The definition must be in one.
    regions: set[int]


@dataclass
class MLIRImporter:
    """Parsed MLIR generic form to Kirin IR.

    Build it over the export's name table and call :meth:`module` once::

        statements = MLIRImporter(NameTable.of(dialects)).module(parse_module(text))

    Args:
        names: the name table the module was emitted with, built over the
            same dialects.
        symbols: the kernel each ``@symbol`` stands for. A reference to a
            kernel not in it is refused.
    """

    names: NameTable
    symbols: Mapping[str, ir.Method] = field(default_factory=dict)
    _values: dict[str, tuple[ir.SSAValue, int]] = field(
        default_factory=dict, init=False
    )
    _forward: dict[str, _Forward] = field(default_factory=dict, init=False)
    _open: list[int] = field(default_factory=list, init=False)
    _defined_in: dict[int, list[str]] = field(default_factory=dict, init=False)
    _labels: list[dict[str, ir.Block]] = field(default_factory=list, init=False)
    _next_region: int = field(default=0, init=False)

    def module(self, op: parse.Operation) -> list[ir.Statement]:
        """The statements of a ``builtin.module``, in order, not in any block.

        Raises:
            MLIRImportError: if anything in it cannot be rebuilt faithfully,
                with the line of the operation it was found on.
        """
        if (
            op.name != "builtin.module"
            or op.operands
            or op.results
            or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1
            or op.regions[0].blocks[0].arguments
        ):
            raise MLIRImportError(
                'expected a "builtin.module" with one region of one block and '
                "nothing else, as the exporter writes it",
                line=op.line,
            )
        (body,) = op.regions[0].blocks
        scope = self._region_scope({})
        statements = [self.statement(child) for child in body.operations]
        self._close(scope)
        self.finish()
        return statements

    def finish(self) -> None:
        """Refuse a value that was used and never defined where it is visible."""
        if self._forward:
            name, forward = min(self._forward.items(), key=lambda item: item[1].line)
            raise MLIRImportError(
                f"{name} is used but never defined in a region that encloses the "
                "use",
                line=forward.line,
            )

    # -- operations ----------------------------------------------------------

    def statement(self, op: parse.Operation) -> ir.Statement:
        """One operation, and everything nested in it, as a Kirin statement."""
        try:
            cls = self.names.statement(op.name)
            if op.properties:
                raise MLIRImportError(
                    "the exporter writes every attribute in the dictionary and "
                    "never as a property (<{...}>)"
                )
            args = [
                self.use(name, typ, op.line)
                for name, typ in zip(op.operands, op.operand_types)
            ]
            args_slice = arguments_slice(cls, len(args), op.attributes)
            regions = [self.region(region) for region in op.regions]
            successors = [self.block(label) for label in op.successors]
            entries = {k: v for k, v in op.attributes.items() if k != SEGMENT_SIZES}
            attributes = import_statement_attributes(cls, entries, self.symbols)
            result_types = list(map(import_type, op.result_types))
            schema = fields(cls)
            _check_count("result", len(schema.results), len(result_types))
            _check_count("region", len(schema.regions), len(regions))
            _check_count("successor", len(schema.blocks), len(successors))

            stmt = cls.__new__(cls)
            ir.Statement.__init__(
                stmt,
                args=args,
                regions=regions,
                successors=successors,
                attributes=attributes,
                result_types=result_types,
                args_slice=args_slice,
            )
            hook = STATEMENT_IMPORTS.get(cls)
            if hook is not None:
                hook(stmt)
            for name, result in zip(op.results, stmt.results):
                self.define(name, result, op.line)
            return stmt
        except MLIRImportError as err:
            raise err.at(op.line)

    # -- regions and blocks ---------------------------------------------------

    def region(self, region: parse.Region) -> ir.Region:
        """A region, its blocks made first so a successor may name a later one."""
        blocks = [ir.Block() for _ in region.blocks]
        labels: dict[str, ir.Block] = {}
        for parsed, block in zip(region.blocks, blocks):
            if parsed.label is None:
                continue
            if parsed.label in labels:
                raise MLIRImportError(
                    f"the block {parsed.label} is defined twice in one region",
                    line=parsed.line,
                )
            labels[parsed.label] = block

        scope = self._region_scope(labels)
        for parsed, block in zip(region.blocks, blocks):
            for name, typ in parsed.arguments:
                try:
                    argument = block.args.append_from(import_type(typ))
                except MLIRImportError as err:
                    raise err.at(parsed.line)
                self.define(name, argument, parsed.line)
            for op in parsed.operations:
                block.stmts.append(self.statement(op))
        self._close(scope)
        return ir.Region(blocks)

    def block(self, label: str) -> ir.Block:
        """The block ``label`` names in the region being read."""
        block = self._labels[-1].get(label) if self._labels else None
        if block is None:
            raise MLIRImportError(
                f"{label} names no block of the region the branch is in; a "
                "successor must be a block of the same region"
            )
        return block

    def _region_scope(self, labels: dict[str, ir.Block]) -> int:
        region = self._next_region
        self._next_region += 1
        self._open.append(region)
        self._defined_in[region] = []
        self._labels.append(labels)
        return region

    def _close(self, region: int) -> None:
        """Leave a region: what it defines is no longer visible."""
        assert self._open.pop() == region, "regions close in the order they open"
        self._labels.pop()
        for name in self._defined_in.pop(region):
            del self._values[name]

    # -- values ----------------------------------------------------------------

    def use(self, name: str, typ: parse.Type, line: int) -> ir.SSAValue:
        """The value ``name`` stands for where it is used as a ``typ``."""
        expected = import_type(typ)
        found = self._values.get(name)
        if found is not None:
            value, _ = found
            _check_type(name, expected, value.type, "defined")
            return value
        forward = self._forward.get(name)
        if forward is None:
            forward = self._forward[name] = _Forward(
                ir.TestValue(expected), line, set(self._open)
            )
        else:
            _check_type(name, expected, forward.placeholder.type, "used before")
            forward.regions &= set(self._open)
        return forward.placeholder

    def define(self, name: str, value: ir.SSAValue, line: int) -> None:
        """Bind ``name`` to ``value`` in the region being read."""
        if name in self._values:
            raise MLIRImportError(
                f"{name} is defined twice where both are visible", line=line
            )
        region = self._open[-1]
        forward = self._forward.pop(name, None)
        if forward is not None:
            if region not in forward.regions:
                raise MLIRImportError(
                    f"{name} is used at line {forward.line}, outside the region "
                    "that defines it",
                    line=line,
                )
            _check_type(name, forward.placeholder.type, value.type, "defined")
            forward.placeholder.replace_by(value)
        # The emitter's hints come back; mlir-opt's %0 and %5#1, and our own
        # numbered names, are not hints, and Kirin's name pattern refuses them.
        hint = name[1:]
        if ir.SSAValue.name_pattern.fullmatch(hint):
            value.name = hint
        self._values[name] = (value, region)
        self._defined_in[region].append(name)


def arguments_slice(
    cls: type[ir.Statement],
    count: int,
    attributes: Mapping[str, parse.Attribute],
) -> dict[str, int | slice]:
    """Which of ``count`` operands each argument field of ``cls`` takes.

    Kirin's ``_name_args_slice``, built as the statement's generated
    ``__init__`` builds it: in field order, an index for a single operand and
    a slice for a group.

    Raises:
        MLIRImportError: if the operands do not divide among the fields, or
            ``operandSegmentSizes`` is missing, malformed or not needed.
    """
    arguments = fields(cls).args
    groups = sum(1 for argument in arguments.values() if argument.group)
    written = attributes.get(SEGMENT_SIZES)

    if written is not None:
        if groups < 2:
            raise MLIRImportError(
                f"{SEGMENT_SIZES} is written only for a statement with two or "
                f"more argument groups, and this one has {groups}"
            )
        if not (
            isinstance(written, parse.DenseArrayAttr)
            and written.element == parse.IntegerType(32)
        ):
            raise MLIRImportError(f"{SEGMENT_SIZES} must be an array<i32: ...>")
        sizes = list(written.values)
        if len(sizes) != len(arguments):
            raise MLIRImportError(
                f"{SEGMENT_SIZES} has {len(sizes)} entries, and the statement "
                f"has {len(arguments)} argument fields"
            )
        for (name, argument), size in zip(arguments.items(), sizes):
            if size < 0 or (not argument.group and size != 1):
                raise MLIRImportError(
                    f"{SEGMENT_SIZES} gives the field {name!r} {size} operands"
                )
    elif groups >= 2:
        raise MLIRImportError(
            f"the statement has {groups} argument groups, so the exporter writes "
            f"{SEGMENT_SIZES}, and it is missing"
        )
    elif not arguments:
        # Statements such as py.tuple.new declare no argument field and take
        # every operand; Kirin then records no slice.
        return {}
    elif groups == 0:
        sizes = [1] * len(arguments)
    else:
        rest = count - (len(arguments) - 1)
        sizes = [rest if argument.group else 1 for argument in arguments.values()]

    if sum(sizes) != count or min(sizes, default=0) < 0:
        raise MLIRImportError(
            f"the statement's argument fields take {sum(sizes)} operands, and "
            f"the operation has {count}"
        )
    position, slices = 0, {}
    for (name, argument), size in zip(arguments.items(), sizes):
        if argument.group:
            slices[name] = slice(position, position + size)
        else:
            slices[name] = position
        position += size
    return slices


def _check_type(
    name: str, expected: ir.Attribute, actual: ir.Attribute, other: str
) -> None:
    """Refuse a use of ``name`` as ``expected`` where it is ``actual``.

    ``mlir-opt`` refuses the same text: every use of a value has its type.
    """
    if expected != actual:
        raise MLIRImportError(
            f"{name} is used as {expected!r} and {other} as {actual!r}; every "
            "use of a value must have its type"
        )


def _check_count(what: str, declared: int, found: int) -> None:
    """Refuse ``found`` results, regions or successors where fields declare others.

    Only where the statement declares some: a hand-written ``__init__`` may
    give a statement what no field declares, as ``py.tuple.new`` does its
    operands.
    """
    if declared and found != declared:
        raise MLIRImportError(
            f"the statement declares {declared} {what}(s), and the operation "
            f"has {found}"
        )
