"""Reproduce the findings recorded in docs/assessment.md.

Run this before trusting an assessment claim that a code change depends on.
Kirin is pre-1.0 and its internals move; the assessment was written against
commit 0c559fd67 (version 0.23.0-dev).

    uv run python scripts/probe_kirin_ir.py

Requires kirin-toolchain to be importable. Nothing here writes to the Kirin
checkout; it only imports and inspects.
"""

from kirin import types
from kirin.prelude import basic, basic_no_opt, structural_no_opt


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


@basic_no_opt
def untyped(x: int, y: int):
    z = x + y
    if z > 0:
        return z
    return -z


@basic(typeinfer=True)
def typed(x: int, y: int) -> int:
    return x + y


@basic_no_opt
def not_inferred(x: int, y: int) -> int:
    return x + y


@basic
def closure(x: int):
    def inner():
        return x

    return inner


@structural_no_opt
def loop(n: int):
    acc = 0
    for i in range(n):
        acc = acc + i
    return acc


def result_types(method):
    return [
        str(r.type)
        for b in method.callable_region.blocks
        for s in b.stmts
        for r in s.results
    ]


def main() -> None:
    rule("Gap 1+5: printed IR is near-MLIR but not MLIR; %self leads every block")
    untyped.code.print()
    entry = untyped.callable_region.blocks[0]
    print("\nentry block args:", [(a.name, str(a.type)) for a in entry.args])
    print("-> first arg is %self of MethodType; it is dropped at the export boundary")

    rule("Gap 2: SSA values are not reliably typed")
    print("basic_no_opt (typeinfer off):", result_types(not_inferred))
    print("basic(typeinfer=True)      :", result_types(typed))
    print("-> export must force TypeInfer; declared TypeVars survive otherwise")

    rule("Gap 1: the type lattice has no MLIR counterpart")
    for t in [
        types.Any,
        types.Bottom,
        types.PyClass(int),
        types.Literal(3),
        types.Union(types.PyClass(int), types.PyClass(str)),
        types.TypeVar("T"),
    ]:
        print(f"  {type(t).__name__:12} {t}")
    print("-> these need the !kirin.* family; none map onto upstream MLIR types")

    rule("Gap 3+4: closures are func.Lambda holding captured values")
    closure.code.print()
    for b in closure.callable_region.blocks:
        for s in b.stmts:
            for k, v in s.attributes.items():
                print(f"  attr {s.dialect.name}.{s.name}.{k}: {type(v).__name__}")
    print("-> lambda lifting is an export precondition")

    rule("Gap 5: Kirin scf.for is not MLIR scf.for")
    loop.code.print()
    print("-> iterates an arbitrary Python iterable, not lb/ub/step")

    rule("Statement shape maps 1:1 onto MLIR Operation")
    stmt = untyped.callable_region.blocks[0].stmts.at(0)
    print(f"  mnemonic : {stmt.dialect.name}.{stmt.name}")
    print(f"  operands : {len(stmt.args)}")
    print(f"  results  : {len(stmt.results)}")
    print(f"  successors: {len(stmt.successors)}")
    print(f"  regions  : {len(stmt.regions)}")
    print(f"  attributes: {dict(stmt.attributes)}")
    print(f"  traits   : {sorted(type(t).__name__ for t in stmt.traits)}")

    rule("External registration works (why this package need not fork Kirin)")
    from kirin.dialects import func

    print("  func dialect interp keys:", sorted(func.dialect.interps))
    print("  Dialect.register mutates this dict, from any package, at import time")


if __name__ == "__main__":
    main()
