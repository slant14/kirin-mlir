"""The guard that keeps this package honest as Kirin moves.

Issue 02's first acceptance criterion: *every* constructor in
``kirin/ir/attrs/types.py`` has either a documented mapping or a documented
refusal, enumerated so that a type added upstream cannot slip through.

Kirin is pre-1.0. If someone adds a lattice element upstream and this package
is not taught about it, the failure without this test is a bad export at some
later date; with it, the failure is here, immediately, with the name of the new
class in the message.
"""

import inspect

from kirin.ir.attrs import types as kirin_types
from kirin.ir.attrs.abc import Attribute

from kirin_mlir.types import TYPE_SCHEMA, TYPE_HANDLERS


def _lattice_classes() -> set[type]:
    """Every concrete ``Attribute`` subclass *defined in* Kirin's types module.

    Filtered on ``__module__`` so re-exports are not counted, and on
    ``isabstract`` so the ``TypeAttribute`` base is not.
    """
    found = set()
    for obj in vars(kirin_types).values():
        if not inspect.isclass(obj) or not issubclass(obj, Attribute):
            continue
        if obj.__module__ != kirin_types.__name__ or inspect.isabstract(obj):
            continue
        found.add(obj)
    return found


def test_every_kirin_lattice_element_is_accounted_for():
    upstream = _lattice_classes()
    handled = set(TYPE_HANDLERS)

    missing = upstream - handled
    assert not missing, (
        "Kirin defines lattice elements this package has never been taught: "
        + ", ".join(sorted(cls.__name__ for cls in missing))
        + " -- add a mapping or a documented refusal in kirin_mlir/types.py"
    )

    stale = handled - upstream
    assert (
        not stale
    ), "kirin_mlir.types maps classes Kirin no longer defines: " + ", ".join(
        sorted(cls.__name__ for cls in stale)
    )


def test_the_expected_inventory_is_the_one_the_design_was_written_against():
    """A change here means the assessment needs re-reading, not just a patch."""
    assert {cls.__name__ for cls in _lattice_classes()} == {
        "AnyType",
        "BottomType",
        "PyClass",
        "Literal",
        "Union",
        "TypeVar",
        "Vararg",
        "Generic",
        "FunctionType",
        "TypeofMethodType",
    }


def test_every_emitted_mnemonic_is_declared_in_the_schema():
    """The IRDL schema (issue 04) and the emitter must not drift apart."""
    from kirin import types
    from kirin.ir.attrs.types import Union, Vararg, Literal

    from kirin_mlir import emit_type

    samples = [
        types.Any,
        types.Bottom,
        types.Int,
        Literal(1),
        Union([types.Int, types.String]),
        Vararg(types.Any),
        types.List,
        types.MethodType[[types.Int], types.Int],
        types.MethodType,
    ]
    emitted = set()
    for sample in samples:
        text = emit_type(sample)
        for mnemonic in TYPE_SCHEMA:
            if f"!kirin.{mnemonic}<" in text or text == f"!kirin.{mnemonic}":
                emitted.add(mnemonic)

    assert emitted == set(TYPE_SCHEMA), set(TYPE_SCHEMA) - emitted
