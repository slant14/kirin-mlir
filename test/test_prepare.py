"""Export preconditions: typed, closure-free, gathered into one module (issue 03)."""

import pytest
from kirin import ir
from kirin.prelude import basic, basic_no_opt
from kirin.dialects import py, func
from kirin.ir.attrs.types import TypeVar

from kirin_mlir import PreparedModule, MLIRExportError, prepare_for_mlir


@basic_no_opt
def untyped(x: int, y: int) -> int:
    return x + y


@basic(typeinfer=True)
def callee(y: int) -> int:
    return y * 2


@basic(typeinfer=True)
def caller(x: int) -> int:
    return callee(x) + 1


@basic(typeinfer=True)
def no_capture(x: int) -> int:
    def inner(y: int) -> int:
        return y + 1

    return inner(x)


@basic(typeinfer=True)
def const_capture(x: int) -> int:
    k = 10

    def inner(y: int) -> int:
        return y + k

    return inner(x)


@basic(typeinfer=True)
def escaping(x: int):
    def inner():
        return x

    return inner


@basic(typeinfer=True)
def local_call(x: int) -> int:
    def inner(y: int) -> int:
        return y + x

    return inner(1)


@basic(typeinfer=True)
def countdown(n: int) -> int:
    if n > 0:
        return countdown(n - 1) + 1
    return 0


@basic(typeinfer=True)
def diamond(x: int) -> int:
    return caller(x) + callee(x)


@basic(typeinfer=True)
def first(x: int) -> int:
    def inner(y: int) -> int:
        return y + 1

    return inner(x)


@basic(typeinfer=True)
def second(x: int) -> int:
    def inner(y: int) -> int:
        return y + 2

    return inner(x)


@basic(typeinfer=True)
def clashing(x: int) -> int:
    return first(x) + second(x)


def _result_types(method: ir.Method) -> list:
    return [r.type for stmt in method.callable_region.walk() for r in stmt.results]


def _symbols(module: PreparedModule) -> list[str]:
    return [method.sym_name for method in module.methods]


def test_an_untyped_kernel_comes_out_typed():
    """``basic_no_opt`` leaves ``~T``; MLIR needs a concrete type on every value."""
    assert any(isinstance(t, TypeVar) for t in _result_types(untyped))

    module = prepare_for_mlir(untyped)

    assert not any(isinstance(t, TypeVar) for t in _result_types(module.entry))
    assert module.entry.inferred


def test_preparing_does_not_change_the_kernel():
    """Exporting must not rewrite the kernel the caller goes on running."""
    before = [str(t) for t in _result_types(untyped)]
    module = prepare_for_mlir(untyped)
    assert module.entry is not untyped
    assert [str(t) for t in _result_types(untyped)] == before


def test_called_kernels_are_gathered_into_the_module():
    module = prepare_for_mlir(caller)
    assert _symbols(module) == ["caller", "callee"]
    invoke = next(
        s for s in module.entry.callable_region.walk() if isinstance(s, func.Invoke)
    )
    assert module.symbol(invoke.callee) == "callee"
    assert module.symbol(callee) == "callee"


def test_a_kernel_reached_twice_is_prepared_once():
    module = prepare_for_mlir(diamond)
    assert sorted(_symbols(module)) == ["callee", "caller", "diamond"]


def test_a_recursive_kernel_is_prepared_once():
    module = prepare_for_mlir(countdown)
    assert _symbols(module) == ["countdown"]


@pytest.mark.parametrize(
    "kernel", [no_capture, const_capture], ids=lambda k: k.sym_name
)
def test_a_nested_function_is_lifted_and_reachable(kernel):
    """Issue 03's acceptance criterion: the lifted symbol is in the module."""
    module = prepare_for_mlir(kernel)
    assert _symbols(module) == [kernel.sym_name, "inner"]
    inner = module.methods[1]
    assert isinstance(inner.code, func.Function)
    assert not any(
        isinstance(s, (func.Lambda, func.GetField))
        for s in inner.callable_region.walk()
    )


def test_a_constant_capture_becomes_a_constant():
    """What Kirin's lambda lifting does: ``getfield(%self, 0)`` becomes ``10``."""
    inner = prepare_for_mlir(const_capture).methods[1]
    constants = [
        s.value.data for s in inner.callable_region.walk() if isinstance(s, py.Constant)
    ]
    assert 10 in constants


def test_the_shared_closure_object_is_left_alone():
    """The callee is promoted in a copy; the original is still a closure."""
    invoke = next(
        s for s in const_capture.callable_region.walk() if isinstance(s, func.Invoke)
    )
    prepare_for_mlir(const_capture)
    assert isinstance(invoke.callee.code, func.Lambda)


@pytest.mark.parametrize("kernel", [escaping, local_call], ids=lambda k: k.sym_name)
def test_a_closure_over_run_time_values_is_refused(kernel):
    with pytest.raises(MLIRExportError, match="known only at run time") as caught:
        prepare_for_mlir(kernel)
    assert isinstance(caught.value.stmt, func.Lambda)


def test_two_kernels_with_one_symbol_name_are_refused():
    """``first`` and ``second`` each hold a different function named ``inner``."""
    with pytest.raises(MLIRExportError, match="both named 'inner'"):
        prepare_for_mlir(clashing)


def test_symbol_lookup_refuses_a_kernel_outside_the_module():
    module = prepare_for_mlir(callee)
    with pytest.raises(MLIRExportError, match="not in this module"):
        module.symbol(caller)


def test_the_module_reports_every_dialect_its_kernels_use():
    module = prepare_for_mlir(caller)
    assert module.dialects == frozenset(caller.dialects.data)
