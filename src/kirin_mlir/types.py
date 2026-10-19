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

The way back (issue 06) is :func:`import_type`, one importer per mnemonic in
:data:`TYPE_IMPORTERS`, kept beside the handlers for the same reason as the
schema. A class comes back through :func:`pyclass_named`, which never imports
a module.
"""

from __future__ import annotations

import sys
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

from . import parse
from .errors import MLIRExportError, MLIRImportError
from .values import escape_string, emit_python_value, import_python_value

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


# -- the way back (issue 06) --------------------------------------------------


def pyclass_named(name: str) -> PyClass:
    """The ``PyClass`` whose class :func:`qualified_name` names ``name``.

    Its inverse, for the import. Looked up first among the classes Kirin has
    made types of in this process: that is Kirin's own interning table, so a
    class comes back with the ``display_name`` and ``prefix`` it was
    registered with, and ``builtins.NoneType``, which no module holds as an
    attribute, is found. Then among the modules already imported, so a class
    that only a kernel's annotations name is found before Kirin has seen it.

    It never imports a module. The text chose the name, and importing a
    module runs that module's code.

    Raises:
        MLIRImportError: if no class loaded in this process has that name.
    """
    # PyClassMeta keeps the table as a private attribute; there is no public
    # view of it. Without it, only the module search below is left.
    for typ, pyclass in list(getattr(PyClass, "_cache", {}).items()):
        if _is_named(typ, name):
            return pyclass
    parts = name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        found: Any = sys.modules.get(".".join(parts[:split]))
        for part in parts[split:]:
            found = getattr(found, part, None)
        if isinstance(found, type) and _is_named(found, name):
            return PyClass(found)
    raise MLIRImportError(
        f"no class named {name!r} is loaded in this process; the import never "
        "imports a module the text names, so import the one that defines it "
        "first"
    )


def _is_named(typ: type, name: str) -> bool:
    try:
        return qualified_name(typ) == name
    except MLIRExportError:
        return False


def import_type(typ: parse.Type) -> TypeAttribute:
    """The Kirin type :func:`emit_type` wrote as ``typ``.

    Its inverse, for the import, and exact but for two losses that are the
    export's, not choices the import could make differently. A ``TypeVar``
    was written as its bound (:func:`_emit_typevar`) and comes back as the
    bound. A ``FunctionType`` whose return type was unspecified (``None``)
    was written as returning ``!kirin.any`` and comes back so, which Kirin
    does not count as equal. A prepared module has neither today.

    Raises:
        MLIRImportError: for anything :func:`emit_type` never writes: a
            builtin MLIR type, another dialect's type, an unknown mnemonic,
            parameters of the wrong number or kind, or ``!kirin.vararg``
            anywhere but last among a ``!kirin.generic``'s arguments.
    """
    result = _import_type(typ)
    if isinstance(result, Vararg):
        raise MLIRImportError(
            f"!{DIALECT}.vararg is a type only as the last argument of a "
            f"!{DIALECT}.generic"
        )
    return result


def _import_type(typ: parse.Type) -> TypeAttribute | Vararg:
    """:func:`import_type`, letting a ``Vararg`` through for the caller to place."""
    if not isinstance(typ, parse.DialectType):
        raise MLIRImportError(
            f"found the builtin type {typ}; the exporter writes every Kirin type "
            f"as a !{DIALECT}.* type"
        )
    if typ.dialect != DIALECT:
        raise MLIRImportError(
            f"found {typ}, which is not a Kirin type; the exporter writes every "
            f"Kirin type as a !{DIALECT}.* type"
        )
    schema = TYPE_SCHEMA.get(typ.name)
    if schema is None:
        raise MLIRImportError(
            f"found {typ}, which is not one of the !{DIALECT}.* types: "
            f"{', '.join(sorted(TYPE_SCHEMA))}"
        )
    if len(typ.params) != len(schema.parameters):
        raise MLIRImportError(
            f"{schema.spelling} takes {len(schema.parameters)} parameter(s), and "
            f"{len(typ.params)} were found"
        )
    return TYPE_IMPORTERS[typ.name](*typ.params)


def _type_parameter(attr: parse.Attribute, mnemonic: str) -> TypeAttribute:
    if not isinstance(attr, parse.TypeAttr):
        raise MLIRImportError(f"a parameter of !{DIALECT}.{mnemonic} must be a type")
    return import_type(attr.type)


def _type_list(
    attr: parse.Attribute, mnemonic: str, *, vararg_last: bool = False
) -> list[TypeAttribute | Vararg]:
    if not isinstance(attr, parse.ArrayAttr) or not all(
        isinstance(element, parse.TypeAttr) for element in attr.elements
    ):
        raise MLIRImportError(
            f"a parameter of !{DIALECT}.{mnemonic} must be an array of types"
        )
    types = [_import_type(element.type) for element in attr.elements]
    for index, typ in enumerate(types):
        if isinstance(typ, Vararg) and not (vararg_last and index == len(types) - 1):
            raise MLIRImportError(
                f"!{DIALECT}.vararg is a type only as the last argument of a "
                f"!{DIALECT}.generic"
            )
    return types


def _import_any() -> AnyType:
    return AnyType()


def _import_bottom() -> BottomType:
    return BottomType()


def _import_pyclass(name: parse.Attribute) -> PyClass:
    if not isinstance(name, parse.StringAttr):
        raise MLIRImportError(
            f"the parameter of !{DIALECT}.pyclass must be the class's qualified "
            "name, as a string"
        )
    return pyclass_named(name.value)


def _import_literal(value: parse.Attribute, typ: parse.Attribute) -> Literal:
    datatype = _type_parameter(typ, "literal")
    data = import_python_value(value, datatype)
    try:
        return Literal(data, datatype)
    except ValueError:
        raise MLIRImportError(
            f"a literal's value must be hashable, and a {type(data).__name__} " "is not"
        ) from None


def _import_union(members: parse.Attribute) -> TypeAttribute:
    # Kirin's Union simplifies on construction, as it did before export, so
    # the members the exporter wrote rebuild the same union.
    types = _type_list(members, "union")
    if len(types) < 2:
        raise MLIRImportError(
            f"!{DIALECT}.union has at least two members; the exporter writes a "
            "union of one as that type and of none as !kirin.bottom"
        )
    return Union(types)


def _import_vararg(element: parse.Attribute) -> Vararg:
    return Vararg(_type_parameter(element, "vararg"))


def _import_generic(body: parse.Attribute, args: parse.Attribute) -> Generic:
    body_type = _type_parameter(body, "generic")
    if not isinstance(body_type, PyClass):
        raise MLIRImportError(
            f"the body of !{DIALECT}.generic must be a !{DIALECT}.pyclass"
        )
    return Generic(body_type, *_type_list(args, "generic", vararg_last=True))


def _import_function_type(
    params: parse.Attribute, result: parse.Attribute
) -> FunctionType:
    return FunctionType(
        tuple(_type_list(params, "method")), _type_parameter(result, "method")
    )


def _import_typeof_method() -> TypeofMethodType:
    return TypeofMethodType()


#: The inverse of :data:`TYPE_HANDLERS`, keyed by mnemonic like
#: :data:`TYPE_SCHEMA`, which fixes how many parameters each importer takes.
#: ``TypeVar`` has no entry, having no mnemonic. ``test/test_import.py`` fails
#: if the keys ever differ from the schema's.
TYPE_IMPORTERS: dict[str, Callable[..., TypeAttribute | Vararg]] = {
    "any": _import_any,
    "bottom": _import_bottom,
    "pyclass": _import_pyclass,
    "literal": _import_literal,
    "union": _import_union,
    "vararg": _import_vararg,
    "generic": _import_generic,
    "method": _import_function_type,
    "anymethod": _import_typeof_method,
}
