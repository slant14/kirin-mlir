"""The ``emit.mlir`` emitter: a :class:`PreparedModule` as MLIR generic form text.

Issue 03. MLIR's generic form prints every operation the same way (ADR 0002,
decision 4)::

    %r = "dialect.op"(%operands) [^successors] ({regions}) {attributes} : (types) -> (types)

so one method, :meth:`EmitMLIR.emit_operation`, reached through Kirin's
``eval_fallback``, emits every statement of every dialect, including dialects
written after this package. A per-dialect method table registered under
``emit.mlir`` is needed only where a statement cannot be emitted that way;
Kirin ``func.func``, whose ``%self`` argument is dropped, is the one case
(:mod:`kirin_mlir.dialects.func`).

What the emitter decides, beyond the template:

**Names.** Operation names come from a :class:`NameTable` built over every
dialect in the module, so a name that is illegal or collides is refused before
any text is written. SSA names keep Kirin's hint where MLIR can spell it
(``%x``, ``%total_1``) and are numbered otherwise. They are unique across the
whole module, not per function: an unregistered op does not isolate its
regions, and real ``mlir-opt`` rejects a function body that reuses a name
defined at module level ("redefinition of SSA value").

**Kernels in attributes.** A ``PyAttr`` whose payload is a Kirin ``Method`` --
``func.invoke``'s callee, or a kernel held by ``py.constant`` -- is emitted as
a symbol reference, ``@callee``, resolved through
:meth:`PreparedModule.symbol`. MLIR resolves calls by symbol, and every kernel
the module reaches is in it.

**Operand groups.** A statement with two or more variadic argument groups
(Kirin ``cf.cond_br``, ``func.call``) carries ``operandSegmentSizes``, one
entry per argument field in operand order, in the attribute dictionary. Never
as properties (``<{...}>``), which IRDL dialects refuse.

**Uses without definitions.** A value used in the module but never defined in
it -- a dropped ``%self`` that something still reads, or a value from outside
the module -- is refused rather than left for ``mlir-opt`` to report as an
undeclared name.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import IO
from dataclasses import field, dataclass
from collections.abc import Sequence, Collection

from kirin import ir
from kirin.decl import fields
from kirin.emit import EmitABC, EmitFrame

from .attrs import emit_statement_attribute
from .names import NameTable
from .types import emit_type
from .errors import MLIRExportError
from .values import escape_string
from .prepare import PreparedModule

#: The attribute MLIR reads operand group sizes from.
SEGMENT_SIZES = "operandSegmentSizes"

#: MLIR's ``bare-id``: an attribute key or symbol name that needs no quoting.
_BARE_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_$.]*")

#: The SSA name hints kept: a Python identifier MLIR can spell. Anything else
#: (a non-ASCII identifier, say) is numbered instead.
_VALUE_HINT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def bare_or_quoted(name: str) -> str:
    """``name`` as MLIR writes an attribute key or a symbol: bare if it can be."""
    if _BARE_ID.fullmatch(name):
        return name
    return f'"{escape_string(name)}"'


@dataclass
class MLIRFrame(EmitFrame[str]):
    """One kernel being emitted.

    SSA and block names are kept on the emitter, not here, because MLIR
    scopes them over the whole module (see the module docstring).
    """

    io: IO[str] = field(kw_only=True)

    def write_line(self, text: str) -> None:
        self.io.write("  " * self._indent + text + "\n")


@dataclass
class EmitMLIR(EmitABC[MLIRFrame, str]):
    """Kirin IR to MLIR generic assembly form.

    Build it over the module's dialects and call :meth:`run` with a
    :class:`PreparedModule`, or use :func:`emit_mlir`, which does both.

    :meth:`operation_name`, :meth:`type_text`, :meth:`attribute_entries` and
    :meth:`result_values` decide what each part of an operation says, and
    :meth:`emit_operation` decides how it is laid out. A subclass that
    overrides the former targets other dialects with the same layout; the
    upstream oracle in the tests does exactly that.
    """

    keys = ("emit.mlir",)
    void = ""

    io: StringIO = field(default_factory=StringIO, init=False)
    module: PreparedModule | None = field(default=None, init=False)
    names: NameTable | None = field(default=None, init=False)
    _values: dict[ir.SSAValue, str] = field(default_factory=dict, init=False)
    _taken: set[str] = field(default_factory=set, init=False)
    _next_value: int = field(default=0, init=False)
    _blocks: dict[ir.Block, str] = field(default_factory=dict, init=False)
    _defined: set[ir.SSAValue] = field(default_factory=set, init=False)
    _first_use: dict[ir.SSAValue, ir.Statement] = field(
        default_factory=dict, init=False
    )

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> MLIRFrame:
        return MLIRFrame(node, io=self.io, has_parent_access=has_parent_access)

    def reset(self) -> None:
        self.io = StringIO()
        self.module = None
        self.names = None
        self._values = {}
        self._taken = set()
        self._next_value = 0
        self._blocks = {}
        self._defined = set()
        self._first_use = {}

    def run(self, node: PreparedModule | ir.Method | ir.Statement) -> str:  # type: ignore[override]
        """Emit a prepared module and return its text.

        Raises:
            MLIRExportError: if ``node`` is not a :class:`PreparedModule`, or
                anything in it has no faithful MLIR spelling.
        """
        if not isinstance(node, PreparedModule):
            raise _unprepared(node)
        # Only a dialect with statements can be emitted (see emit_mlir).
        missing = {d for d in node.dialects if d.stmts} - self.dialects.data
        if missing:
            raise MLIRExportError(
                "this emitter was built without the dialects "
                f"{sorted(d.name for d in missing)}, which the module uses; "
                "build it over PreparedModule.dialects"
            )

        self.reset()
        self.module = node
        self.names = NameTable.of(node.dialects)
        with self.eval_context():
            self.io.write('"builtin.module"() ({\n')
            for method in node.methods:
                with self.new_frame(method.code) as frame, frame.indent():
                    frame.current_stmt = method.code
                    self.frame_eval(frame, method.code)
            self.io.write("}) : () -> ()\n")
        self._check_definitions()
        return self.io.getvalue()

    def eval_fallback(self, frame: MLIRFrame, node: ir.Statement) -> tuple[str, ...]:
        """Every statement without a method table of its own."""
        return self.emit_operation(frame, node)

    # -- what each part says -------------------------------------------------

    def operation_name(self, node: ir.Statement) -> str:
        """The quoted name: the mangled ``kirin_*`` name (ADR 0002)."""
        assert self.names is not None, "only called while a module is emitted"
        return self.names.operation(node)

    def type_text(self, typ: ir.Attribute) -> str:
        """A value's type: the ``!kirin.*`` spelling of its lattice type."""
        return emit_type(typ)

    def attribute_entries(self, node: ir.Statement) -> dict[str, str]:
        """The attribute dictionary, keyed as MLIR will key it."""
        entries = {
            key: self.attribute_text(value, stmt=node, key=key)
            for key, value in node.attributes.items()
        }
        sizes = segment_sizes(node)
        if sizes is not None:
            if SEGMENT_SIZES in entries:
                raise MLIRExportError(
                    f"the statement has its own attribute {SEGMENT_SIZES!r}, "
                    "which MLIR reserves for operand group sizes",
                    stmt=node,
                )
            entries[SEGMENT_SIZES] = f"array<i32: {', '.join(map(str, sizes))}>"
        return entries

    def result_values(self, node: ir.Statement) -> Sequence[ir.SSAValue]:
        """The values the operation defines."""
        return node.results

    # -- how an operation is laid out -----------------------------------------

    def emit_operation(
        self,
        frame: MLIRFrame,
        node: ir.Statement,
        *,
        omit: Collection[ir.SSAValue] = (),
    ) -> tuple[str, ...]:
        """Write ``node`` in generic form and return its result names.

        Args:
            frame: the frame being written to.
            node: the statement.
            omit: block arguments of ``node``'s regions to leave out. Anything
                that still uses one is refused when the module is finished.
        """
        try:
            results = [self.define(value) for value in self.result_values(node)]
            operands = [self.use(value, node) for value in node.args]
            signature = (
                f"({', '.join(self.type_text(v.type) for v in node.args)}) -> "
                f"({', '.join(self.type_text(v.type) for v in self.result_values(node))})"
            )
        except MLIRExportError as err:
            raise err.attach(stmt=node)

        head = f"{', '.join(results)} = " if results else ""
        text = f'{head}"{self.operation_name(node)}"({", ".join(operands)})'
        if node.successors:
            text += f" [{', '.join(self.block_name(b) for b in node.successors)}]"

        entries = self.attribute_entries(node)
        attributes = (
            " {"
            + ", ".join(f"{bare_or_quoted(k)} = {v}" for k, v in entries.items())
            + "}"
            if entries
            else ""
        )
        tail = f"{attributes} : {signature}"

        if not node.regions:
            frame.write_line(text + tail)
            return tuple(results)

        frame.write_line(text + " ({")
        for index, region in enumerate(node.regions):
            if index:
                frame.write_line("}, {")
            self.emit_region(frame, node, region, omit)
        frame.write_line("})" + tail)
        return tuple(results)

    def emit_region(
        self,
        frame: MLIRFrame,
        owner: ir.Statement,
        region: ir.Region,
        omit: Collection[ir.SSAValue],
    ) -> None:
        for block in region.blocks:
            frame.current_block = block
            try:
                params = ", ".join(
                    f"{self.define(arg)}: {self.type_text(arg.type)}"
                    for arg in block.args
                    if arg not in omit
                )
            except MLIRExportError as err:
                raise err.attach(stmt=owner)
            label = self.block_name(block)
            frame.write_line(f"{label}({params}):" if params else f"{label}:")
            with frame.indent():
                for stmt in block.stmts:
                    frame.current_stmt = stmt
                    self.frame_eval(frame, stmt)

    # -- names -------------------------------------------------------------

    def attribute_text(
        self, value: ir.Attribute, *, stmt: ir.Statement, key: str
    ) -> str:
        """One attribute value; a kernel becomes a reference to its symbol."""
        if isinstance(value, ir.PyAttr) and isinstance(value.data, ir.Method):
            assert self.module is not None, "only called while a module is emitted"
            try:
                return "@" + bare_or_quoted(self.module.symbol(value.data))
            except MLIRExportError as err:
                raise err.attach(stmt=stmt, key=key)
        return emit_statement_attribute(stmt, key, value)

    def define(self, value: ir.SSAValue) -> str:
        """Name a value at its definition."""
        self._defined.add(value)
        return self.value_name(value)

    def use(self, value: ir.SSAValue, user: ir.Statement) -> str:
        """Name a value where ``user`` reads it."""
        self._first_use.setdefault(value, user)
        return self.value_name(value)

    def value_name(self, value: ir.SSAValue) -> str:
        """``%hint``, made unique across the module, or ``%N``.

        A use may come before its definition in the text, which MLIR allows,
        so a name is fixed the first time a value is seen either way.
        """
        name = self._values.get(value)
        if name is not None:
            return name
        hint = value.name
        if hint is not None and _VALUE_HINT.fullmatch(hint):
            name, suffix = hint, 0
            while name in self._taken:
                suffix += 1
                name = f"{hint}_{suffix}"
        else:
            # Hints start with a letter or underscore, so a bare number can
            # never collide with one.
            name = str(self._next_value)
            self._next_value += 1
        self._taken.add(name)
        self._values[value] = name = f"%{name}"
        return name

    def block_name(self, block: ir.Block) -> str:
        name = self._blocks.get(block)
        if name is None:
            name = self._blocks[block] = f"^bb{len(self._blocks)}"
        return name

    def _check_definitions(self) -> None:
        for value, user in self._first_use.items():
            if value not in self._defined:
                raise MLIRExportError(
                    f"the value {self._values[value]} is used but not defined "
                    "anywhere in the module. It is either a dropped %self that "
                    "something still reads (ADR 0001, decision 2) or a value "
                    "from outside the module",
                    stmt=user,
                )


def segment_sizes(stmt: ir.Statement) -> list[int] | None:
    """``operandSegmentSizes`` for ``stmt``, or ``None`` if it needs none.

    Needed when two or more argument fields are variadic, because MLIR cannot
    otherwise tell where one group ends (ADR 0002, decision 4). One entry per
    argument field, in operand order, so a single operand counts 1.
    """
    schema = fields(type(stmt))
    arguments = {**schema.std_args, **schema.kw_args}
    if sum(1 for arg in arguments.values() if arg.group) < 2:
        return None

    # Kirin records which operands belong to which field. That is a private
    # attribute, but the public views over it (``Statement.args`` and the
    # named properties) cannot say where an empty group sits.
    spans: list[tuple[int, int, str]] = []
    for name, index in stmt._name_args_slice.items():
        if isinstance(index, int):
            spans.append((index, index + 1, name))
        else:
            spans.append((index.start, index.stop, name))
    spans.sort(key=lambda span: span[:2])

    position, sizes = 0, []
    for start, stop, _ in spans:
        if start != position or stop < start:
            break
        sizes.append(stop - start)
        position = stop
    if position != len(stmt.args) or {name for *_, name in spans} != set(arguments):
        raise MLIRExportError(
            "cannot compute operandSegmentSizes: the statement's argument "
            "fields do not divide its operands into consecutive groups",
            stmt=stmt,
        )
    return sizes


def emit_mlir(module: PreparedModule) -> str:
    """The MLIR generic form text of a prepared module.

    The output is one ``builtin.module`` holding a ``kirin_func.func`` per
    kernel, in the order :func:`prepare_for_mlir` found them::

        text = emit_mlir(prepare_for_mlir(kernel))

    Raises:
        MLIRExportError: if ``module`` is not a :class:`PreparedModule` -- an
            ``ir.Method`` must go through :func:`prepare_for_mlir` first -- or
            anything in it has no faithful MLIR spelling.
    """
    if not isinstance(module, PreparedModule):
        raise _unprepared(module)
    # A dialect with no statements, such as lowering.cf, only teaches Kirin's
    # Python front end, which emitting does not use. Building a group also
    # builds that front end, and it refuses two dialects that lower the same
    # syntax: a @structural kernel calling a @basic one brings scf and
    # lowering.cf, which both lower ``for``. So those dialects are left out.
    statements = [dialect for dialect in module.dialects if dialect.stmts]
    return EmitMLIR(ir.DialectGroup(statements)).run(module)


def _unprepared(node: object) -> MLIRExportError:
    return MLIRExportError(
        f"cannot emit {node!r} directly: MLIR needs a concrete type on every "
        "value, no closures, and every called kernel in one module, which "
        "Kirin does not guarantee. Call kirin_mlir.prepare_for_mlir(method) "
        "and emit what it returns"
    )
