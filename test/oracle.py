"""Driving real ``mlir-opt`` as a subprocess.

ADR 0001, decision 4: real MLIR is the oracle. Tests assert validity by handing
text to ``mlir-opt`` and checking that it parses, never by comparing against
expected text.

Until issue 04 there is no IRDL, so ``!kirin.*`` types and every emitted
``kirin_*`` operation belong to *unregistered* dialects. MLIR still parses
those under ``--allow-unregistered-dialect``, storing them opaquely. That is a
weaker check than verification -- it proves the text is well-formed MLIR, not
that it means anything -- but it is a real one for the builtin attributes,
which are parsed for real, and for the structure of every operation. Issue 03
adds :func:`mlir_opt` for the stronger checks: ``--verify-roundtrip``, and the
upstream oracle in ``upstream_oracle.py``, which is verified in full.
"""

from __future__ import annotations

import shutil
import subprocess

MLIR_OPT = shutil.which("mlir-opt")

SKIP_REASON = (
    "mlir-opt not found on PATH; install it from apt.llvm.org (see README "
    "prerequisites). Real MLIR is the oracle, so these checks cannot be "
    "faked with a reimplementation."
)


def mlir_opt(text: str, *flags: str) -> subprocess.CompletedProcess:
    """Run ``mlir-opt`` with ``flags`` over ``text`` and return the process."""
    assert MLIR_OPT is not None
    return subprocess.run(
        [MLIR_OPT, *flags],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )


def parses(text: str) -> subprocess.CompletedProcess:
    """Run ``mlir-opt`` over ``text`` and return the completed process."""
    return mlir_opt(text, "--allow-unregistered-dialect")


def module_with_types(type_texts: list[str]) -> str:
    """A minimal module whose only content produces values of those types."""
    results = ", ".join(type_texts)
    return (
        '"builtin.module"() ({\n'
        f'  %0:{len(type_texts)} = "test.produce"() : () -> ({results})\n'
        "}) : () -> ()\n"
    )


def module_with_attributes(attributes: dict[str, str]) -> str:
    """A minimal module whose only operation carries those attributes."""
    body = ", ".join(f"{key} = {value}" for key, value in attributes.items())
    return (
        '"builtin.module"() ({\n'
        f'  "test.carry"() {{{body}}} : () -> ()\n'
        "}) : () -> ()\n"
    )
