"""The ``emit.mlir`` emitter, checked by real ``mlir-opt`` (issue 03).

Two oracles. Every kernel's own emission must survive
``mlir-opt --allow-unregistered-dialect --verify-roundtrip``: until issue 04
registers the leaf dialects, that proves well-formed structure, not meaning.
The arithmetic slice is also emitted through ``UpstreamOracle``
(``upstream_oracle.py``), as upstream ``arith``/``cf``/``func``, which
``mlir-opt`` verifies in full -- types, terminators, successor arity, and that
every called symbol exists with the right type.
"""

import re

import pytest
from kirin import ir
from oracle import MLIR_OPT, SKIP_REASON, parses, mlir_opt
from kirin.prelude import basic, structural, basic_no_opt
from kirin.dialects import cf, py, func
from upstream_oracle import KeepsSelf, UpstreamOracle

from kirin_mlir import (
    EmitMLIR,
    PreparedModule,
    MLIRExportError,
    emit_mlir,
    emit_attribute,
    prepare_for_mlir,
)
from kirin_mlir.emit import segment_sizes

needs_mlir = pytest.mark.skipif(MLIR_OPT is None, reason=SKIP_REASON)


@basic(typeinfer=True)
def callee(y: int) -> int:
    return y * 2


@basic(typeinfer=True)
def branches(x: int, y: int) -> int:
    z = x + y
    if z > 0:
        return callee(z)
    return z - 1


@basic(typeinfer=True)
def countdown(n: int) -> int:
    if n > 0:
        return countdown(n - 1) + 1
    return 0


@basic_no_opt
def untyped(x: int, y: int) -> int:
    return x + y


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
def apply(f: ir.Method[[int], int], x: int) -> int:
    return f(x)


@basic(typeinfer=True)
def higher_order(x: int) -> int:
    return apply(callee, x)


@basic(typeinfer=True)
def negate(x: int) -> int:
    return -x


@basic(typeinfer=True)
def iterates(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i
    return total


@basic(typeinfer=True)
def containers(x: int):
    xs = [x, x]
    return xs[0] + x


@structural(typeinfer=True)
def structured_if(x: int) -> int:
    if x > 0:
        y = x + 1
    else:
        y = x - 1
    return y


@structural(typeinfer=True)
def structured_for(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i
    return total


#: Kernels inside the oracle's slice: integer arithmetic and comparison,
#: constants, Kirin func and cf.
SLICE = [branches, countdown, untyped, no_capture, const_capture, higher_order]

#: Everything, including dialects with no upstream counterpart.
EVERY = [*SLICE, negate, iterates, containers, structured_if, structured_for]

#: Every name the emitter writes must be this, bar the module wrapper.
MANGLED = re.compile(r"kirin_[a-z0-9_]+\.[a-z0-9_]+")


def ids(kernel: ir.Method) -> str:
    return kernel.sym_name


def upstream(kernel: ir.Method, emitter: type[EmitMLIR] = UpstreamOracle) -> str:
    module = prepare_for_mlir(kernel)
    return emitter(ir.DialectGroup(module.dialects)).run(module)


# -- the emitter's own output -------------------------------------------------


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_kernel_round_trips(kernel):
    """Issue 03's first criterion, before issue 04: unregistered, round-tripped."""
    text = emit_mlir(prepare_for_mlir(kernel))
    result = mlir_opt(text, "--allow-unregistered-dialect", "--verify-roundtrip")
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


@needs_mlir
@pytest.mark.parametrize("kernel", EVERY, ids=ids)
def test_every_operation_is_in_a_kirin_namespace(kernel):
    """No op can land in an upstream namespace; the wrapper is the only one.

    The names are read back from ``mlir-opt``'s own generic printout, so this
    checks what MLIR parsed, not what the emitter meant to write.
    """
    text = emit_mlir(prepare_for_mlir(kernel))
    result = mlir_opt(text, "--allow-unregistered-dialect", "--mlir-print-op-generic")
    assert result.returncode == 0, result.stderr
    names = re.findall(r'"([\w.$]+)"\(', result.stdout)
    assert names.count("builtin.module") == 1
    strays = [n for n in names if n != "builtin.module" and not MANGLED.fullmatch(n)]
    assert not strays


@needs_mlir
def test_the_signature_spelling_parses():
    signature = emit_attribute(branches.code.attributes["signature"])
    result = parses(f'"test.carry"() {{signature = {signature}}} : () -> ()\n')
    assert result.returncode == 0, result.stderr


# -- the upstream oracle ------------------------------------------------------


@needs_mlir
@pytest.mark.parametrize("kernel", SLICE, ids=ids)
def test_the_slice_verifies_as_upstream_mlir(kernel):
    """Issue 03's third criterion: arith, cf and func, fully verified.

    No ``--allow-unregistered-dialect``: every op is registered, so MLIR
    checks types, terminators, branch arity and every call's target.
    """
    text = upstream(kernel)
    result = mlir_opt(text, "--verify-roundtrip")
    assert result.returncode == 0, f"{result.stderr}\n--- input ---\n{text}"


@needs_mlir
def test_the_oracle_rejects_a_kept_self():
    """The negative control: without the %self rule the entry block is wrong."""
    result = mlir_opt(upstream(branches, KeepsSelf))
    assert result.returncode != 0
    assert "match function signature" in result.stderr


@needs_mlir
def test_an_untyped_kernel_exports_with_concrete_types():
    """Issue 03's fourth criterion, through the oracle.

    ``basic_no_opt`` leaves ``~T``. The oracle has no spelling for an unknown
    type, so verifying at all means every value came out as a concrete int.
    """
    result = mlir_opt(upstream(untyped), "--verify-roundtrip")
    assert result.returncode == 0, result.stderr


@needs_mlir
@pytest.mark.parametrize("kernel", [no_capture, const_capture], ids=ids)
def test_a_lifted_nested_function_is_a_callable_symbol(kernel):
    """Issue 03's fifth criterion: MLIR resolves ``func.call @inner``."""
    text = upstream(kernel)
    assert "@inner" in text
    result = mlir_opt(text, "--verify-roundtrip")
    assert result.returncode == 0, result.stderr


@needs_mlir
def test_a_kernel_held_as_a_constant_is_a_symbol_reference():
    """``py.constant Method("callee")`` becomes ``@callee``, and MLIR checks it.

    Upstream ``func.constant`` verifies that the symbol names a function of
    the constant's type, so a wrong or missing reference would be rejected.
    """
    module = prepare_for_mlir(higher_order)
    constant = next(
        stmt
        for stmt in module.entry.callable_region.walk()
        if isinstance(stmt, py.constant.Constant)
    )
    assert isinstance(constant.value, ir.PyAttr)
    assert isinstance(constant.value.data, ir.Method)
    result = mlir_opt(upstream(higher_order), "--verify-roundtrip")
    assert result.returncode == 0, result.stderr


# -- refusals -----------------------------------------------------------------


def test_an_unprepared_method_is_refused():
    with pytest.raises(MLIRExportError, match="prepare_for_mlir"):
        emit_mlir(untyped)  # type: ignore[arg-type]


def test_the_emitter_itself_refuses_an_unprepared_method():
    emitter = EmitMLIR(untyped.dialects)
    with pytest.raises(MLIRExportError, match="prepare_for_mlir"):
        emitter.run(untyped)


def test_an_emitter_missing_a_dialect_is_refused():
    module = prepare_for_mlir(branches)
    emitter = EmitMLIR(ir.DialectGroup([func.dialect]))
    with pytest.raises(MLIRExportError, match="built without the dialects"):
        emitter.run(module)


def test_a_function_that_reads_self_is_refused():
    module = prepare_for_mlir(callee)
    entry = module.entry.code.body.blocks[0]
    read = func.GetField(entry.args[0], field=0)
    assert entry.first_stmt is not None
    read.insert_before(entry.first_stmt)
    with pytest.raises(MLIRExportError, match="still reads %self") as caught:
        emit_mlir(module)
    assert caught.value.stmt is module.entry.code


def test_a_closure_that_bypassed_preparation_is_refused():
    """A module not built by ``prepare_for_mlir`` can still carry a closure."""
    invoke = next(
        stmt
        for stmt in const_capture.callable_region.walk()
        if isinstance(stmt, func.Invoke)
    )
    closure = invoke.callee
    assert isinstance(closure.code, func.Lambda)
    module = PreparedModule((closure,), (closure,))
    with pytest.raises(MLIRExportError, match=r"func\.lambda"):
        emit_mlir(module)


def test_a_use_with_no_definition_in_the_module_is_refused():
    """A value from outside the module would reach MLIR undeclared."""
    module = prepare_for_mlir(callee)
    outsider = prepare_for_mlir(branches).entry.code.body.blocks[0].args[1]
    ret = next(
        stmt
        for stmt in module.entry.callable_region.walk()
        if isinstance(stmt, func.Return)
    )
    ret.args = (outsider,)
    with pytest.raises(MLIRExportError, match="not defined anywhere") as caught:
        emit_mlir(module)
    assert caught.value.stmt is ret


# -- operand groups -----------------------------------------------------------


def test_segment_sizes_cover_every_argument_field():
    """``cf.cond_br`` is (cond, then_arguments, else_arguments)."""
    module = prepare_for_mlir(branches)
    cond_br = next(
        stmt
        for stmt in module.entry.callable_region.walk()
        if isinstance(stmt, cf.ConditionalBranch)
    )
    sizes = segment_sizes(cond_br)
    assert sizes is not None
    assert len(sizes) == 3 and sizes[0] == 1
    assert sum(sizes) == len(cond_br.args)


def test_one_variadic_group_needs_no_segment_sizes():
    module = prepare_for_mlir(branches)
    invoke = next(
        stmt
        for stmt in module.entry.callable_region.walk()
        if isinstance(stmt, func.Invoke)
    )
    assert segment_sizes(invoke) is None
