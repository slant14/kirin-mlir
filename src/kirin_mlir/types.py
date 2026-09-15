"""Kirin's type lattice, mapped onto the ``!kirin.*`` MLIR type family.

Kirin's types (``kirin/ir/attrs/types.py``) form a lattice built for abstract
interpretation: ``Union``, ``Literal``, ``TypeVar``, ``AnyType``, ``BottomType``,
``Generic``, ``PyClass``, ``FunctionType``, ``TypeofMethodType``. MLIR has no
counterpart for most of them, so ``docs/assessment.md`` concludes that a
dedicated ``!kirin.*`` family is required and that no mapping onto upstream
MLIR types is faithful in the general case.

Two constraints shape the spellings below.

**Every type has fixed arity.** IRDL (``irdl.parameters``) declares a *fixed*
number of parameters per type; it has no variadic parameters. So every
naturally variadic piece -- a union's members, a generic's arguments, a
method's parameter list -- is carried as a **single** parameter holding an
MLIR ``ArrayAttr``. That keeps issue 04 able to declare these types in IRDL at
all, instead of discovering the problem after the emitter is written.

**Emission is deterministic.** ``Union.types`` is a ``frozenset``. Iterating it
gives an order that varies with ``PYTHONHASHSEED``, which would make emitted
text differ between runs of the same program. Members are therefore sorted by
their emitted text.

The family::

    !kirin.any                                   AnyType
    !kirin.bottom                                BottomType
    !kirin.pyclass<"module.Qualname">            PyClass
    !kirin.literal<value, type>                  Literal
    !kirin.union<[t, ...]>                       Union
    !kirin.vararg<t>                             Vararg
    !kirin.generic<body, [t, ...]>               Generic
    !kirin.method<[t, ...], ret>                 FunctionType
    !kirin.anymethod                             TypeofMethodType

``TypeVar`` has no spelling of its own -- see :func:`_emit_typevar`.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

from kirin.ir.attrs.abc import Attribute
from kirin.ir.attrs.types import (
    Union,
    Vararg,
    AnyType,
    Generic,
    Literal,
    PyClass,
    TypeVar,
    BottomType,
    FunctionType,
    TypeAttribute,
    TypeofMethodType,
)

from .errors import MLIRExportError
from .values import escape_string, emit_python_value

#: The MLIR dialect namespace these types live in. Kirin's own ``func``, ``cf``
#: and ``scf`` are emitted out-of-tree for the same reason (ADR 0001,
#: decision 3): the names collide with upstream MLIR, the semantics do not.
DIALECT = "kirin"


@dataclass(frozen=True)
class TypeSchema:
    """Shape of one ``!kirin.*`` type, for issue 04 to generate IRDL from.

    ``parameters`` names each IRDL parameter and the constraint it needs, in
    order. An empty tuple means a plain nominal type with no parameters.
    """

    mnemonic: str
    parameters: tuple[str, ...]
    summary: str

    @property
    def spelling(self) -> str:
        return f"!{DIALECT}.{self.mnemonic}"


#: The declarative half of this module. Issue 04 turns this into IRDL; the
#: handler table below turns Kirin values into text that conforms to it. They
#: are kept side by side so the two cannot drift.
TYPE_SCHEMA: dict[str, TypeSchema] = {
    "any": TypeSchema(
        "any",
        (),
        "a value whose type is unknown; the honest spelling of Kirin's AnyType",
    ),
    "bottom": TypeSchema(
        "bottom",
        (),
        "the empty type; no value inhabits it, so it marks unreachable code",
    ),
    "pyclass": TypeSchema(
        "pyclass",
        ("name: StringAttr, the fully qualified 'module.Qualname'",),
        "a nominal Python class",
    ),
    "literal": TypeSchema(
        "literal",
        ("value: AnyAttr, a builtin attribute", "type: AnyType, the value's type"),
        "a singleton type inhabited by exactly one compile-time value",
    ),
    "union": TypeSchema(
        "union",
        ("members: ArrayAttr of types, sorted, at least two",),
        "a join of two or more types",
    ),
    "vararg": TypeSchema(
        "vararg",
        ("element: AnyType",),
        "zero or more of the element type; only valid inside a parameter list",
    ),
    "generic": TypeSchema(
        "generic",
        ("body: !kirin.pyclass", "args: ArrayAttr of types"),
        "a parameterised Python class such as list[int] or tuple[int, ...]",
    ),
    "method": TypeSchema(
        "method",
        ("params: ArrayAttr of types", "result: AnyType"),
        "the type of a Kirin Method with a known signature",
    ),
    "anymethod": TypeSchema(
        "anymethod",
        (),
        "the type of some Kirin Method whose signature is not known",
    ),
}


def qualified_name(typ: type) -> str:
    """The cross-process identity of a Python class: ``module.Qualname``.

    ``PyClass`` also carries ``display_name`` and ``prefix``, which is what
    makes Kirin print ``!py.IList``. Those are *not* independent information:
    ``PyClassMeta.__call__`` (``kirin/ir/attrs/types.py``) caches one ``PyClass``
    per Python class and raises if a second registration supplies a different
    display name or prefix. They are a function of ``typ``, recoverable from a
    dialect's ``python_types`` registry, so the qualified name alone is a
    complete identity and the sugar is not emitted.
    """
    module = getattr(typ, "__module__", None)
    qualname = getattr(typ, "__qualname__", None) or getattr(typ, "__name__", None)
    if not module or not qualname:
        raise MLIRExportError(
            f"cannot name the Python class {typ!r}: it has no __module__ or "
            "__qualname__, so it has no identity outside this process"
        )
    return f"{module}.{qualname}"


def pyclass_type(typ: type) -> str:
    """Emit ``!kirin.pyclass<"module.Qualname">`` for a Python class."""
    return f'!{DIALECT}.pyclass<"{escape_string(qualified_name(typ))}">'


def emit_type(attr: Attribute) -> str:
    """Emit one Kirin type attribute as ``!kirin.*`` MLIR type text.

    Raises:
        MLIRExportError: if the lattice element has no faithful spelling. This
            fires for any ``TypeAttribute`` subclass this package has not been
            taught, including one added upstream after it was written.
    """
    for cls in type(attr).__mro__:
        handler = TYPE_HANDLERS.get(cls)
        if handler is not None:
            return handler(attr)
    raise MLIRExportError(
        f"no MLIR type mapping for the Kirin lattice element "
        f"{type(attr).__name__} ({attr!r}); kirin_mlir.types must be taught "
        "how to spell it, or taught to refuse it for a stated reason"
    )


def _emit_any(_: AnyType) -> str:
    return f"!{DIALECT}.any"


def _emit_bottom(_: BottomType) -> str:
    # A value typed Bottom cannot exist, so its presence means the region is
    # unreachable or inference contradicted itself. That is worth carrying into
    # MLIR exactly, not erasing: !kirin.bottom is precise, not a widening.
    return f"!{DIALECT}.bottom"


def _emit_pyclass(attr: PyClass) -> str:
    return pyclass_type(attr.typ)


def _emit_literal(attr: Literal) -> str:
    # Literal keeps a declared type alongside the datum, because the Python
    # type of the datum need not be the type in the IR -- the class docstring's
    # own example is Literal(1, types.Int32). Both parameters are emitted.
    return (
        f"!{DIALECT}.literal<{emit_python_value(attr.data)}, "
        f"{emit_type(attr.type)}>"
    )


def _emit_union(attr: Union) -> str:
    # Sorted, because ``types`` is a frozenset: iteration order depends on
    # PYTHONHASHSEED, and unsorted output would differ run to run for the same
    # program. Kirin collapses the degenerate cases before we ever see them --
    # Union([]) constructs a BottomType and Union([t]) constructs t -- so a
    # Union reaching here always has at least two members.
    members = sorted(emit_type(member) for member in attr.types)
    return f"!{DIALECT}.union<[{', '.join(members)}]>"


def _emit_typevar(attr: TypeVar) -> str:
    """A ``TypeVar`` is emitted as its bound.

    MLIR has no type variables: by the time IR is exported, every value needs a
    concrete type. A ``TypeVar`` that survived ``PrepareForMLIR``'s type
    inference is one inference could not resolve, and the only sound thing left
    to say about the value is its bound.

    This is Kirin's own reading, not an invention: ``TypeVar`` delegates
    subtyping to its bound (``is_subseteq_fallback`` in
    ``kirin/ir/attrs/types.py`` returns ``self.bound.is_subseteq(other)``). The
    default bound is ``AnyType``, so the common unresolved ``~T`` becomes
    ``!kirin.any`` exactly as issue 02 requires, while a bounded variable keeps
    the constraint instead of being flattened away with it.
    """
    return emit_type(attr.bound)


def _emit_vararg(attr: Vararg) -> str:
    return f"!{DIALECT}.vararg<{emit_type(attr.typ)}>"


def _emit_generic(attr: Generic) -> str:
    # Kirin stores the trailing vararg in its own slot rather than in ``vars``.
    # Folding it back into the argument list as a !kirin.vararg keeps this type
    # at fixed arity -- body plus one array -- which is what IRDL can declare.
    args = [emit_type(var) for var in attr.vars]
    if attr.vararg is not None:
        args.append(emit_type(attr.vararg))
    body = emit_type(attr.body)
    return f"!{DIALECT}.generic<{body}, [{', '.join(args)}]>"


def _emit_function_type(attr: FunctionType) -> str:
    params = ", ".join(emit_type(param) for param in attr.params_type)
    if attr.return_type is None:
        # "Unspecified", not "returns nothing" -- Kirin spells the latter as
        # PyClass(NoneType). Unknown is exactly what !kirin.any means.
        result = f"!{DIALECT}.any"
    else:
        result = emit_type(attr.return_type)
    return f"!{DIALECT}.method<[{params}], {result}>"


def _emit_typeof_method(_: TypeofMethodType) -> str:
    # types.MethodType: "some method", with no signature. It gets its own
    # mnemonic rather than a zero-parameter !kirin.method because IRDL types
    # have fixed arity, so one name cannot have both 0 and 2 parameters.
    return f"!{DIALECT}.anymethod"


#: Every ``Attribute`` subclass defined in ``kirin/ir/attrs/types.py`` that is
#: not abstract appears here exactly once. ``test/test_lattice_coverage.py``
#: enumerates that module and fails if the two sets ever diverge, so a type
#: added upstream cannot slip through silently.
TYPE_HANDLERS: dict[type, Callable[[Any], str]] = {
    AnyType: _emit_any,
    BottomType: _emit_bottom,
    PyClass: _emit_pyclass,
    Literal: _emit_literal,
    Union: _emit_union,
    TypeVar: _emit_typevar,
    Vararg: _emit_vararg,
    Generic: _emit_generic,
    FunctionType: _emit_function_type,
    TypeofMethodType: _emit_typeof_method,
}


def is_type_attribute(attr: Attribute) -> bool:
    """True for things :func:`emit_type` handles, including ``Vararg``.

    ``Vararg`` is a plain ``Attribute`` upstream, not a ``TypeAttribute``, even
    though it only ever appears inside one.
    """
    return isinstance(attr, (TypeAttribute, Vararg))
