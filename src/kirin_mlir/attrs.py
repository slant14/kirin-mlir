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
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from kirin import ir
from kirin.ir.attrs.types import AnyType, PyClass

from .types import emit_type, is_type_attribute
from .errors import MLIRExportError
from .values import emit_python_value

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


def emit_statement_attributes(stmt: ir.Statement) -> dict[str, str]:
    """Emit every attribute of one statement, keyed as MLIR will key them.

    This is the frame that knows the context an error needs, so it is the one
    that supplies it.
    """
    return {
        key: emit_attribute(value, stmt=stmt, key=key)
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
