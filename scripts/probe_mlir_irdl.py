"""Reproduce what real MLIR accepts, recorded in docs/assessment.md.

The companion to ``probe_kirin_ir.py``: that script checks claims about Kirin,
this one checks claims about MLIR. Every finding in the assessment's
"real MLIR" addendum and in ``docs/adr/0002-irdl-scope-and-naming.md`` is a
case below. Run it before trusting one of those claims, and again whenever the
installed ``mlir-opt`` changes version -- IRDL is still moving upstream.

    uv run python scripts/probe_mlir_irdl.py

Finds ``mlir-opt`` from ``$MLIR_OPT``, then ``PATH``, then Homebrew's keg-only
``llvm``. Writes only to a temporary directory. Exits non-zero if any finding
no longer reproduces, so a newer MLIR that lifts a limit is noticed.
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def find_mlir_opt() -> str:
    candidates = [os.environ.get("MLIR_OPT"), shutil.which("mlir-opt")]
    brew = shutil.which("brew")
    if brew:
        prefix = subprocess.run(
            [brew, "--prefix", "llvm"], capture_output=True, text=True, check=False
        ).stdout.strip()
        candidates.append(str(Path(prefix) / "bin" / "mlir-opt"))
    for path in candidates:
        if path and Path(path).is_file():
            return path
    sys.exit("mlir-opt not found; see README prerequisites")


MLIR_OPT = find_mlir_opt()
WORK = Path(tempfile.mkdtemp(prefix="probe_mlir_irdl_"))
FAILURES: list[str] = []

INT = '!kirin.pyclass<"builtins.int">'

#: The leaf-dialect IRDL from ADR 0002: mangled names, separate constraint
#: variables where values are independent, one shared variable where Kirin's
#: own declaration shares a TypeVar (BinOp's lhs, rhs and result are all T).
LEAF_IRDL = """
irdl.dialect @kirin {
  irdl.type @any
  irdl.type @pyclass {
    %0 = irdl.base "#builtin.string"
    irdl.parameters(name: %0)
  }
}
irdl.dialect @kirin_py_constant {
  irdl.operation @constant {
    %0 = irdl.any
    %1 = irdl.any
    irdl.attributes {"value" = %0}
    irdl.results(result: %1)
  }
}
irdl.dialect @kirin_py_binop {
  irdl.operation @add {
    %0 = irdl.any
    irdl.operands(lhs: %0, rhs: %0)
    irdl.results(result: %0)
  }
}
"""


def write(name: str, text: str) -> Path:
    path = WORK / name
    path.write_text(text)
    return path


def run(*args: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [MLIR_OPT, *map(str, args)], capture_output=True, text=True, check=False
    )


def first_error(result: subprocess.CompletedProcess) -> str:
    for line in result.stderr.splitlines():
        if "error:" in line:
            return line.split("error:", 1)[1].strip()
    return result.stderr.strip().splitlines()[0] if result.stderr.strip() else ""


def expect(
    title: str,
    result: subprocess.CompletedProcess,
    *,
    accepted: bool,
    error: str | None = None,
    output: str | None = None,
) -> None:
    """Record whether ``result`` matches the documented behaviour."""
    ok = (result.returncode == 0) == accepted
    if ok and error is not None:
        ok = error in result.stderr
    if ok and output is not None:
        ok = output in result.stdout
    seen = "accepted" if result.returncode == 0 else f"rejected: {first_error(result)}"
    print(f"  [{'ok' if ok else 'CHANGED'}] {title}\n           {seen}")
    if not ok:
        FAILURES.append(title)


def irdl_names() -> None:
    rule("1. IRDL names: lowercase letters, digits and underscores only")
    cases = [
        (
            "operation name with a dot is rejected",
            "@py { irdl.operation @binop.add { } }",
            False,
            "must contain only lowercase letters, digits and underscores",
        ),
        (
            "dialect name with a dot is rejected",
            "@py.binop { irdl.operation @add { } }",
            False,
            "invalid dialect name",
        ),
        (
            "uppercase operation name is rejected",
            "@stim { irdl.operation @CX { } }",
            False,
            "should not contain uppercase letters",
        ),
        (
            "uppercase dialect name is rejected",
            "@Py { irdl.operation @add { } }",
            False,
            "should not contain uppercase letters",
        ),
        (
            "a leading underscore is rejected",
            "@kirin_x { irdl.operation @_x { } }",
            False,
            "should not contain leading or double underscores",
        ),
        (
            "a double underscore is rejected",
            "@kirin__x { irdl.operation @add { } }",
            False,
            "should not contain leading or double underscores",
        ),
        (
            "a dialect name starting with a digit is rejected",
            '@"2py" { irdl.operation @add { } }',
            False,
            "invalid dialect name",
        ),
        (
            "underscores are accepted",
            "@kirin_py_binop { irdl.operation @const_none { } }",
            True,
            None,
        ),
    ]
    for title, body, accepted, error in cases:
        expect(
            title,
            run(write("names.irdl.mlir", f"irdl.dialect {body}\n")),
            accepted=accepted,
            error=error,
        )


def leaf_verification() -> None:
    rule("2. Tier 1 for leaf dialects: IRDL really verifies")
    irdl = write("leaf.irdl.mlir", LEAF_IRDL)
    good = write(
        "good.mlir",
        (
            f'%0 = "kirin_py_constant.constant"() {{value = 1 : i64}} : () -> {INT}\n'
            f'%1 = "kirin_py_binop.add"(%0, %0) : ({INT}, {INT}) -> {INT}\n'
            f'%2 = "kirin_py_binop.add"(%0, %0) : ({INT}, {INT}) -> {INT}\n'
        ),
    )
    expect(
        "a well-formed program verifies and round-trips",
        run(f"--irdl-file={irdl}", "--verify-roundtrip", good),
        accepted=True,
    )
    expect(
        "without --irdl-file the same text is rejected as unregistered",
        run(good),
        accepted=False,
        error="unregistered dialect",
    )
    cases = {
        "wrong operand count is rejected": (
            (
                f'%0 = "kirin_py_constant.constant"() {{value = 1 : i64}} : () -> {INT}\n'
                f'%1 = "kirin_py_binop.add"(%0, %0, %0) : ({INT}, {INT}, {INT}) -> {INT}\n'
            ),
            "expects exactly 2 operands",
        ),
        "a malformed !kirin.pyclass parameter is rejected": (
            (
                '%0 = "kirin_py_constant.constant"() {value = 1 : i64}'
                " : () -> !kirin.pyclass<42 : i64>\n"
            ),
            "expected base attribute 'builtin.string'",
        ),
        "a missing attribute is rejected": (
            f'%0 = "kirin_py_constant.constant"() : () -> {INT}\n',
            'attribute "value" is expected but not provided',
        ),
    }
    for title, (text, error) in cases.items():
        expect(
            title,
            run(f"--irdl-file={irdl}", write("bad.mlir", text)),
            accepted=False,
            error=error,
        )

    shared = write(
        "shared.irdl.mlir",
        LEAF_IRDL.replace("    %1 = irdl.any\n", "").replace(
            "results(result: %1)", "results(result: %0)"
        ),
    )
    expect(
        "reusing one constraint variable means 'equal', like a TypeVar",
        run(f"--irdl-file={shared}", good),
        accepted=False,
        error="expected '1 : i64' but got",
    )


def no_traits() -> None:
    rule("3. Tier 2: IRDL declares no traits, so passes cannot transform")
    irdl = WORK / "leaf.irdl.mlir"
    good = WORK / "good.mlir"
    for flag in ("--cse", "--canonicalize"):
        result = run(f"--irdl-file={irdl}", flag, good)
        adds = result.stdout.count("kirin_py_binop.add")
        title = f"{flag} leaves both identical, unused adds in place"
        ok = result.returncode == 0 and adds == 2
        print(f"  [{'ok' if ok else 'CHANGED'}] {title}\n           {adds} adds remain")
        if not ok:
            FAILURES.append(title)


def control_flow() -> None:
    rule("4. Terminators, successors and unknown ops under IRDL")
    irdl = write(
        "cf.irdl.mlir",
        """
irdl.dialect @kirin_func {
  irdl.operation @function {
    %r = irdl.region
    irdl.regions(body: %r)
  }
  irdl.operation @return {
    %0 = irdl.any
    irdl.operands(value: %0)
  }
}
irdl.dialect @kirin_cf {
  irdl.operation @br { }
}
""",
    )
    one_block = write(
        "one_block.mlir",
        """"kirin_func.function"() ({
^bb0(%x: i64):
  "kirin_func.return"(%x) : (i64) -> ()
}) : () -> ()
""",
    )
    expect(
        "an IRDL op cannot be a terminator, even in a one-block function",
        run(f"--irdl-file={irdl}", one_block),
        accepted=False,
        error="block with no terminator",
    )
    branch = write(
        "branch.mlir",
        """"kirin_func.function"() ({
^bb0(%x: i64):
  "kirin_cf.br"()[^bb1] : () -> ()
^bb1:
  "kirin_func.return"(%x) : (i64) -> ()
}) : () -> ()
""",
    )
    expect(
        "an IRDL op cannot carry successors",
        run(f"--irdl-file={irdl}", branch),
        accepted=False,
        error="successors in non-terminator",
    )
    partial = write(
        "partial.irdl.mlir",
        "irdl.dialect @kirin_func {\n  irdl.operation @function {\n"
        "    %r = irdl.region\n    irdl.regions(body: %r)\n  }\n}\n",
    )
    expect(
        "a registered dialect refuses ops it does not declare",
        run(f"--irdl-file={partial}", "--allow-unregistered-dialect", one_block),
        accepted=False,
        error="does not allow unknown operations",
    )
    repeat_irdl = write(
        "repeat.irdl.mlir",
        """
irdl.dialect @kirin_stim_gate {
  irdl.operation @h { }
}
irdl.dialect @kirin_stim_cf {
  irdl.operation @repeat {
    %r = irdl.region
    irdl.regions(body: %r)
  }
}
""",
    )
    body = '"{op}"() ({{\n  "kirin_stim_gate.h"() : () -> ()\n}}) : () -> ()\n'
    expect(
        "a registered region op whose body ends in a registered op is rejected",
        run(
            f"--irdl-file={repeat_irdl}",
            "--allow-unregistered-dialect",
            write("rep1.mlir", body.format(op="kirin_stim_cf.repeat")),
        ),
        accepted=False,
        error="block with no terminator",
    )
    expect(
        "the same body under an unregistered region op is accepted",
        run(
            f"--irdl-file={repeat_irdl}",
            "--allow-unregistered-dialect",
            write("rep2.mlir", body.format(op="kirin_stim_loop.repeat")),
        ),
        accepted=True,
    )
    hybrid = write(
        "hybrid.mlir",
        f""""kirin_func.function"() ({{
^bb0(%x: {INT}):
  %0 = "kirin_py_constant.constant"() {{value = 1 : i64}} : () -> {INT}
  %1 = "kirin_py_binop.add"(%x, %0) : ({INT}, {INT}) -> {INT}
  "kirin_func.return"(%1) : ({INT}) -> ()
}}) : () -> ()
""",
    )
    expect(
        "hybrid: leaf ops registered, control flow unregistered, verifies",
        run(
            f"--irdl-file={WORK / 'leaf.irdl.mlir'}",
            "--allow-unregistered-dialect",
            "--verify-roundtrip",
            hybrid,
        ),
        accepted=True,
    )
    bad_hybrid = write(
        "bad_hybrid.mlir",
        hybrid.read_text().replace(
            f'"kirin_py_binop.add"(%x, %0) : ({INT}, {INT})',
            f'"kirin_py_binop.add"(%x) : ({INT})',
        ),
    )
    expect(
        "hybrid still rejects a malformed leaf op",
        run(
            f"--irdl-file={WORK / 'leaf.irdl.mlir'}",
            "--allow-unregistered-dialect",
            bad_hybrid,
        ),
        accepted=False,
        error="expects exactly 2 operands",
    )


def variadics() -> None:
    rule("5. Two variadic operand groups need operandSegmentSizes, as an attribute")
    irdl = write(
        "call.irdl.mlir",
        """
irdl.dialect @kirin_func {
  irdl.operation @call {
    %0 = irdl.any
    irdl.operands(inputs: variadic %0, kwargs: variadic %0)
    irdl.results(result: %0)
  }
}
""",
    )
    head = '%a = "test.a"() : () -> i64\n%r = "kirin_func.call"(%a, %a, %a) '
    flags = (f"--irdl-file={irdl}", "--allow-unregistered-dialect")
    expect(
        "without segment sizes it is rejected",
        run(*flags, write("c1.mlir", head + ": (i64, i64, i64) -> i64\n")),
        accepted=False,
        error="'operandSegmentSizes' attribute is expected",
    )
    expect(
        "segment sizes in the attribute dictionary are accepted",
        run(
            *flags,
            write(
                "c2.mlir",
                head
                + "{operandSegmentSizes = array<i32: 2, 1>} : (i64, i64, i64) -> i64\n",
            ),
        ),
        accepted=True,
    )
    expect(
        "segment sizes as properties <{...}> are rejected",
        run(
            *flags,
            write(
                "c3.mlir",
                head
                + "<{operandSegmentSizes = array<i32: 2, 1>}> : (i64, i64, i64) -> i64\n",
            ),
        ),
        accepted=False,
        error="extensible Dialects don't support properties",
    )


def syntax_order() -> None:
    rule("6. Generic form: successors come before regions")
    body = '"test.a"() ({{\n^bb0:\n  "test.op"(){} : () -> ()\n^bb1:\n  "test.end"() : () -> ()\n}}) : () -> ()\n'
    expect(
        "[^bb1] ({...}) parses",
        run(
            "--allow-unregistered-dialect",
            write("o1.mlir", body.format("[^bb1] ({\n  })")),
        ),
        accepted=True,
    )
    expect(
        "({...}) [^bb1] does not parse",
        run(
            "--allow-unregistered-dialect",
            write("o2.mlir", body.format(" ({\n  })[^bb1]")),
        ),
        accepted=False,
        error="expected ':'",
    )


def name_clash() -> None:
    rule("7. An IRDL dialect named like an upstream one is accepted silently")
    irdl = write(
        "clash.irdl.mlir",
        "irdl.dialect @math {\n  irdl.operation @sqrt {\n"
        "    %0 = irdl.any\n    irdl.operands(x: %0)\n    irdl.results(r: %0)\n  }\n}\n",
    )
    text = write(
        "clash.mlir",
        '%a = "test.a"() : () -> f64\n%r = "math.sqrt"(%a) : (f64) -> f64\n',
    )
    expect(
        "no error is raised for redefining 'math'",
        run(f"--irdl-file={irdl}", "--allow-unregistered-dialect", text),
        accepted=True,
    )


def survey() -> None:
    rule("8. Survey: which Kirin (and stim) names and shapes these limits affect")
    import pkgutil
    import importlib
    import collections

    from kirin import ir
    from kirin.decl import fields

    upstream = set(
        run("--show-dialects")
        .stdout.split(":", 1)[-1]
        .strip()
        .replace(" ", "")
        .split(",")
    )
    packages = [importlib.import_module("kirin.dialects")]
    try:
        packages.append(importlib.import_module("bloqade.stim"))
    except ImportError:
        print("  (bloqade-circuit not installed; surveying Kirin only)")

    legal = re.compile(r"^[a-z0-9_]+$")
    for pkg in packages:
        # bloqade.stim's package also holds Kirin's `debug` and a pass helper
        # dialect; only the `stim.*` dialects are the payload.
        own = "stim." if pkg.__name__ == "bloqade.stim" else ""
        dialects: dict[str, ir.Dialect] = {}
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            # Never import private modules. kirin/dialects/math/_gen.py is a
            # code generator that rewrites files in the installed package when
            # imported. Each Dialect object is re-exported by its public
            # package, so nothing is lost by skipping `_dialect` and friends.
            if any(part.startswith("_") for part in info.name.split(".")):
                continue
            try:
                module = importlib.import_module(info.name)
            except ImportError:
                # Some dialect modules need optional packages such as numpy;
                # the survey covers whatever this environment can import.
                continue
            for value in vars(module).values():
                if isinstance(value, ir.Dialect) and value.name.startswith(own):
                    dialects[value.name] = value
        illegal, multi, mangled = [], [], collections.defaultdict(list)
        control: set[str] = set()
        dotted = [n for n in dialects if "." in n]
        total = 0
        for name, dialect in sorted(dialects.items()):
            for stmt in dialect.stmts:
                total += 1
                full = f"{name}.{stmt.name}"
                if not legal.match(stmt.name):
                    illegal.append(full)
                schema = fields(stmt)
                args = {**schema.std_args, **schema.kw_args}
                if sum(1 for a in args.values() if a.group) > 1:
                    multi.append(full)
                is_term = any(type(t).__name__ == "IsTerminator" for t in stmt.traits)
                if schema.blocks or schema.regions or is_term:
                    control.add(name)
                key = (
                    name.replace(".", "_").lower(),
                    stmt.name.replace(".", "_").lower(),
                )
                mangled[key].append(full)
        clash = sorted(n for n in dialects if n.replace(".", "_").lower() in upstream)
        print(f"  {pkg.__name__}: {len(dialects)} dialects, {total} statements")
        print(f"    dialect names containing '.' ({len(dotted)} of {len(dialects)})")
        print(
            f"    statement names with '.' or uppercase ({len(illegal)}): {illegal[:8]}"
            f"{' ...' if len(illegal) > 8 else ''}"
        )
        print(f"    statements needing operandSegmentSizes: {multi}")
        print(
            "    dialects IRDL cannot register (a statement has a region, successors"
            f" or IsTerminator): {sorted(control)}"
        )
        print(
            f"    dialects IRDL can register: {len(dialects) - len(control)} of {len(dialects)}"
        )
        print(f"    dialect names that collide with upstream MLIR: {clash}")
        print(
            "    collisions after mangling: "
            f"{[v for v in mangled.values() if len(v) > 1] or 'none'}"
        )


def main() -> None:
    print(f"mlir-opt: {MLIR_OPT}")
    print(run("--version").stdout.strip().splitlines()[0])
    irdl_names()
    leaf_verification()
    no_traits()
    control_flow()
    variadics()
    syntax_order()
    name_clash()
    survey()
    rule("Result")
    if FAILURES:
        print(
            "These findings no longer reproduce -- re-read ADR 0002 before relying on it:"
        )
        for title in FAILURES:
            print(f"  - {title}")
        sys.exit(1)
    print("Every finding reproduces.")


if __name__ == "__main__":
    main()
