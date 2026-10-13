"""``kirin-mlir``: the command line.

Generate IRDL for some of Kirin's dialects, then let ``mlir-opt`` verify a
module emitted by :func:`kirin_mlir.emit_mlir` against it::

    kirin-mlir irdl py.binop py.constant py.cmp -o kirin.irdl.mlir
    mlir-opt --irdl-file=kirin.irdl.mlir --allow-unregistered-dialect module.mlir

A dialect that IRDL cannot register (``func``, ``cf``, ``scf``, ``module``) is
left out and reported on stderr with the statement responsible. Asking for one
by name exits with status 1, so a script notices. ``--all`` takes every
dialect found and only reports the exclusions.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from collections.abc import Sequence

from .irdl import generate_irdl
from .errors import MLIRExportError
from .discover import KIRIN_DIALECTS, find_dialects


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kirin-mlir",
        description="MLIR interoperability for Kirin.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    irdl = commands.add_parser(
        "irdl",
        help="generate IRDL for Kirin dialects",
        description=(
            "Generate IRDL for Kirin dialects, for mlir-opt --irdl-file. Only "
            "dialects with no region, successor or terminator can be "
            "registered; any other is left out and reported."
        ),
    )
    irdl.add_argument(
        "dialects",
        nargs="*",
        metavar="DIALECT",
        help="Kirin dialect names, such as py.binop",
    )
    irdl.add_argument("--all", action="store_true", help="every dialect found")
    irdl.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="PACKAGE",
        help=f"also look for dialects in PACKAGE (always: {KIRIN_DIALECTS})",
    )
    irdl.add_argument(
        "-o", "--output", type=Path, help="write the IRDL here, not to stdout"
    )
    args = parser.parse_args(argv)

    known = find_dialects([KIRIN_DIALECTS, *args.package])
    if args.all:
        chosen = list(known.values())
    elif not args.dialects:
        parser.error("name at least one dialect, or pass --all")
    else:
        unknown = [name for name in args.dialects if name not in known]
        if unknown:
            print(
                f"kirin-mlir: unknown dialect {', '.join(map(repr, unknown))}; "
                f"known: {', '.join(known)}",
                file=sys.stderr,
            )
            return 2
        chosen = [known[name] for name in args.dialects]

    try:
        generated = generate_irdl(chosen)
    except MLIRExportError as err:
        print(f"kirin-mlir: {err}", file=sys.stderr)
        return 1

    if args.output is None:
        sys.stdout.write(generated.text)
    else:
        args.output.write_text(generated.text)

    for excluded in generated.excluded:
        print(
            f"kirin-mlir: not registering {excluded.dialect}: {excluded}, which "
            "IRDL cannot declare (ADR 0002, decision 2)",
            file=sys.stderr,
        )
    return 1 if generated.excluded and not args.all else 0
