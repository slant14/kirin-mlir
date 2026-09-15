"""Optional pure-Python smoke check of the emitted text.

**xDSL is not the oracle.** ADR 0001 decision 4: only real ``mlir-opt`` decides
whether emitted text is valid MLIR, because the claim this project makes is
that *real* MLIR accepts Kirin's IR. This module exists only so that, on a
machine without an LLVM install, gross syntax errors still surface early. It is
skipped when xDSL is absent, and a pass here proves nothing that
``test_mlir_oracle.py`` does not prove better.
"""

import pytest
from test_mlir_oracle import EVERY_TYPE, EVERY_PAYLOAD

from kirin_mlir import emit_type, emit_python_value

xdsl_parser = pytest.importorskip("xdsl.parser")
xdsl_context = pytest.importorskip("xdsl.context")
xdsl_exceptions = pytest.importorskip("xdsl.utils.exceptions")


def _parse(text: str) -> None:
    ctx = xdsl_context.Context(allow_unregistered=True)
    from xdsl.dialects.builtin import Builtin

    ctx.load_dialect(Builtin)
    xdsl_parser.Parser(ctx, text).parse_module()


def test_types_are_syntactically_well_formed():
    for kirin_type in EVERY_TYPE:
        text = emit_type(kirin_type)
        _parse(f'%0 = "test.produce"() : () -> {text}\n')


def test_payloads_are_syntactically_well_formed():
    for payload in EVERY_PAYLOAD:
        text = emit_python_value(payload)
        _parse(f'"test.carry"() {{a = {text}}} : () -> ()\n')


@pytest.mark.parametrize(
    "bad",
    [
        # unbalanced brackets in a parameterised type
        '%0 = "test.produce"() : () -> !kirin.union<[!kirin.any>\n',
        # a float whose mantissa has no '.', which is why _emit_float adds one
        '"test.carry"() {a = 1e+300 : f64} : () -> ()\n',
        '"test.carry"() {a = ???} : () -> ()\n',
    ],
)
def test_the_smoke_check_actually_rejects_something(bad):
    """A parser that accepts everything would make the checks above vacuous."""
    with pytest.raises(xdsl_exceptions.ParseError):
        _parse(bad)
