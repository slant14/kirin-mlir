"""Kirin attributes to MLIR attributes.

A Kirin ``Statement`` carries ``dict[str, Attribute]``, structurally the same
as an MLIR operation's attribute dictionary (``kirin/ir/nodes/stmt.py``). What
differs is what may live in the values.

Three kinds are handled here:

``PyAttr``
    Wraps an arbitrary Python object. Only Kirin's own
    ``SUPPORTED_PYTHON_TYPES`` subset can leave the process at all, and of that
    subset only the payloads with an unambiguous builtin MLIR counterpart cross
    -- see :mod:`kirin_mlir.values`.

``TypeAttribute`` (and ``Vararg``)
    A lattice type used in attribute position. MLIR writes a type where an
    attribute is expected as a ``TypeAttr``, so this is just the type text.

everything else
    Refused. Dialect-specific attributes such as Kirin ``func``'s ``Signature``
    are *not* registered here on purpose: they belong to the dialect's emitter
    method table (issue 03), registered the same way Kirin registers its own
    per-dialect behaviour. :func:`register_attribute` is that door. Keeping a
    general mapper free of imports from particular dialects is what lets this
    package emit dialects it has never seen.

The way back (issue 06) is :func:`import_statement_attribute`. The text does
not say which Kirin attribute class it was written from; the statement's
field declaration does, so every attribute is read against it, and a dialect
attribute class gets its importer through :func:`register_attribute_import`.
"""

from __future__ import annotations

from typing import Any, get_origin
from collections.abc import Mapping, Callable

from kirin import ir
from kirin.decl import fields
from kirin.ir.attrs.types import AnyType, PyClass, FunctionType, TypeAttribute

from . import parse
from .types import emit_type, import_type, is_type_attribute
from .errors import MLIRExportError, MLIRImportError
from .values import emit_python_value, import_python_value

#: Class-keyed handlers, resolved along the MRO. Populated by
#: :func:`register_attribute`.
ATTRIBUTE_HANDLERS: dict[type, Callable[[Any], str]] = {}


def register_attribute(cls: type) -> Callable[[Callable[[Any], str]], Callable]:
    """Register an emitter for a Kirin attribute class.

    The extension point for dialects that define their own attributes. Lookup
    walks the MRO, so registering a base class covers its subclasses.
    """

    def wrapper(fn: Callable[[Any], str]) -> Callable[[Any], str]:
        ATTRIBUTE_HANDLERS[cls] = fn
        return fn

    return wrapper


def emit_attribute(
    attr: ir.Attribute,
    *,
    stmt: ir.Statement | None = None,
    key: str | None = None,
) -> str:
    """Emit one Kirin attribute as MLIR attribute text.

    Args:
        attr: the attribute to emit.
        stmt: the statement it was found on, used only to make an error legible.
        key: the attribute key it was found under, likewise.

    Raises:
        MLIRExportError: naming ``stmt`` and ``key`` when they were supplied.
    """
    try:
        for cls in type(attr).__mro__:
            handler = ATTRIBUTE_HANDLERS.get(cls)
            if handler is not None:
                return handler(attr)
        raise MLIRExportError(
            f"no MLIR attribute mapping for {type(attr).__name__}; register one "
            "with kirin_mlir.attrs.register_attribute if this dialect's "
            "attribute has a faithful MLIR spelling"
        )
    except MLIRExportError as err:
        # Re-raise the same object so the traceback still points at the mapping
        # function that refused, with the caller's context filled in.
        raise err.attach(stmt=stmt, key=key)


def emit_statement_attribute(stmt: ir.Statement, key: str, value: ir.Attribute) -> str:
    """Emit the attribute ``stmt`` holds under ``key``.

    Differs from :func:`emit_attribute` in one case, which needs the
    statement: a ``PyAttr`` whose declared type is exactly the type the
    statement's own definition gives ``key``. ``func.func``'s ``slots`` is
    ``PyAttr(("x", "y"), pytype=tuple[str, ...])`` because its field is
    declared ``slots: tuple[str, ...] = info.attribute()``. That declared type
    is not carried by the payload, but it is not lost either: it is part of
    the operation's definition, which issue 04 generates IRDL from and issue 06
    imports against. So it is dropped, and only the payload is emitted. A
    declared type the schema does not account for still goes through
    :func:`emit_pyattr`, which refuses it.
    """
    if isinstance(value, ir.PyAttr):
        field = fields(type(stmt)).attributes.get(key)
        if field is not None and value.type == field.type:
            try:
                return emit_python_value(value.data)
            except MLIRExportError as err:
                raise err.attach(stmt=stmt, key=key)
    return emit_attribute(value, stmt=stmt, key=key)


def emit_statement_attributes(stmt: ir.Statement) -> dict[str, str]:
    """Emit every attribute of one statement, keyed as MLIR will key them.

    This is the frame that knows the context an error needs, so it is the one
    that supplies it.
    """
    return {
        key: emit_statement_attribute(stmt, key, value)
        for key, value in stmt.attributes.items()
    }


@register_attribute(ir.PyAttr)
def emit_pyattr(attr: ir.PyAttr) -> str:
    """Emit a ``PyAttr`` as the builtin MLIR attribute for its payload.

    ``PyAttr`` carries a payload *and* a type. By default the type is derived
    -- ``PyClass(type(data))``, per ``PyAttr.__init__`` -- and so is recoverable
    from the payload alone; emitting the bare builtin attribute loses nothing.

    An explicitly supplied ``pytype`` that says something the payload does not
    is a different matter. ``PyAttr(1, pytype=SomeInt32)`` emitted as ``1 : i64``
    would quietly become an i64, which is exactly the "verifies but means
    something else" failure ADR 0001 refuses. So that case is rejected rather
    than flattened. ``AnyType`` is allowed through because it asserts nothing.
    """
    data_type = type(attr.data)
    declared = attr.type
    derived = isinstance(declared, PyClass) and declared.typ is data_type
    if not (derived or isinstance(declared, AnyType)):
        raise MLIRExportError(
            f"PyAttr declares type {declared} but carries a "
            f"{data_type.__name__} payload; a builtin MLIR attribute can carry "
            "the payload or the declared type but not both, and dropping the "
            "declared type would change what the attribute means"
        )
    return emit_python_value(attr.data)


@register_attribute(ir.Attribute)
def emit_attribute_fallback(attr: ir.Attribute) -> str:
    """Types in attribute position; refusal for anything else.

    Registered on ``ir.Attribute`` rather than on ``TypeAttribute`` because
    ``Vararg`` is a plain ``Attribute`` upstream despite only ever appearing
    inside a type.
    """
    if is_type_attribute(attr):
        return emit_type(attr)
    raise MLIRExportError(
        f"no MLIR attribute mapping for {type(attr).__name__}; register one "
        "with kirin_mlir.attrs.register_attribute if this dialect's attribute "
        "has a faithful MLIR spelling"
    )


# -- the way back (issue 06) --------------------------------------------------

#: Importers for dialect attribute classes, keyed by the class a statement
#: field declares and resolved along its MRO. Populated by
#: :func:`register_attribute_import`.
ATTRIBUTE_IMPORTERS: dict[type, Callable[[parse.Attribute], ir.Attribute]] = {}


def register_attribute_import(
    cls: type,
) -> Callable[[Callable[[parse.Attribute], ir.Attribute]], Callable]:
    """Register an importer for a Kirin attribute class.

    The inverse of :func:`register_attribute`: a dialect that registers how
    its attribute is written registers here how it is read back.
    """

    def wrapper(
        fn: Callable[[parse.Attribute], ir.Attribute],
    ) -> Callable[[parse.Attribute], ir.Attribute]:
        ATTRIBUTE_IMPORTERS[cls] = fn
        return fn

    return wrapper


def import_statement_attribute(
    stmt: type[ir.Statement],
    key: str,
    attr: parse.Attribute,
    symbols: Mapping[str, ir.Method] | None = None,
) -> ir.Attribute:
    """The attribute a ``stmt`` held under ``key``, read back from ``attr``.

    The inverse of :func:`emit_statement_attribute`, and read against the
    statement's declaration as that function wrote against it:

    - A field declared with a Python type (``slots: tuple[str, ...]``) gets
      ``PyAttr(payload, declared type)``, as the statement's generated
      ``__init__`` builds it. The declared type decides ``tuple`` or
      ``list``, ``bytes`` or ``bytearray``.
    - A field declared ``PyAttr`` or ``Data`` (``py.constant``'s ``value``)
      gets ``PyAttr(payload)``, typed by its payload as ``PyAttr`` types it.
    - A field declared as a ``TypeAttribute`` gets the type.
    - A field declared as any other attribute class gets the importer
      registered for it: Kirin ``func``'s ``Signature`` has one, in
      :mod:`kirin_mlir.dialects.func`.

    A kernel, written ``@symbol``, is looked up in ``symbols``. A field
    declared with a Python type gives it that type (``func.invoke``'s
    ``callee``). Under ``Data`` it gets ``FunctionType(arg_types,
    return_type)``, the rule Kirin's ``py.Constant`` uses (issue 06, step 0,
    decision 7).

    Raises:
        MLIRImportError: naming ``stmt`` and ``key``, if the statement
            declares no such attribute or ``attr`` is not what the exporter
            writes for the declaration.
    """
    try:
        field = fields(stmt).attributes.get(key)
        if field is None:
            raise MLIRImportError(
                "the statement declares no attribute by that name, and the "
                "import reads every attribute against its declaration"
            )
        if field.pytype:
            return ir.PyAttr(_payload(attr, field.type, symbols), field.type)

        declared = get_origin(field.annotation) or field.annotation
        for cls in declared.__mro__:
            importer = ATTRIBUTE_IMPORTERS.get(cls)
            if importer is not None:
                return importer(attr)
        if declared in (ir.Data, ir.PyAttr):
            payload = _payload(attr, AnyType(), symbols)
            if isinstance(payload, ir.Method):
                kernel = FunctionType(payload.arg_types, payload.return_type)
                return ir.PyAttr(payload, kernel)
            return ir.PyAttr(payload)
        if issubclass(declared, TypeAttribute):
            if not isinstance(attr, parse.TypeAttr):
                raise MLIRImportError("the field declares a type, and none was found")
            typ = import_type(attr.type)
            if not isinstance(typ, declared):
                raise MLIRImportError(
                    f"the field declares a {declared.__name__}, and "
                    f"{type(typ).__name__} was found"
                )
            return typ
        raise MLIRImportError(
            f"no import mapping for {declared.__name__}; register one with "
            "kirin_mlir.attrs.register_attribute_import, as for its export"
        )
    except MLIRImportError as err:
        raise err.attach(stmt=stmt, key=key)


def import_statement_attributes(
    stmt: type[ir.Statement],
    entries: Mapping[str, parse.Attribute],
    symbols: Mapping[str, ir.Method] | None = None,
) -> dict[str, ir.Attribute]:
    """Every attribute of one ``stmt``, read back from its dictionary.

    The inverse of :func:`emit_statement_attributes`. ``operandSegmentSizes``
    is not an attribute of the statement, so the caller removes it first.
    """
    return {
        key: import_statement_attribute(stmt, key, attr, symbols)
        for key, attr in entries.items()
    }


def _payload(
    attr: parse.Attribute,
    hint: TypeAttribute,
    symbols: Mapping[str, ir.Method] | None,
) -> Any:
    """A ``PyAttr``'s payload: a kernel for ``@symbol``, else a Python value."""
    if not isinstance(attr, parse.SymbolRefAttr):
        return import_python_value(attr, hint)
    if attr.nested:
        raise MLIRImportError(
            f"@{attr.root} is a nested symbol reference; the exporter writes a "
            "kernel as a flat @symbol"
        )
    method = (symbols or {}).get(attr.root)
    if method is None:
        raise MLIRImportError(f"@{attr.root} names no kernel of the module")
    return method
