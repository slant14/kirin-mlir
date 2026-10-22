# kirin-mlir

kirin-mlir connects [Kirin](https://github.com/QuEraComputing/kirin) to [MLIR](https://mlir.llvm.org). It exports Kirin programs as MLIR, has real `mlir-opt` verify them, and reads them back into Kirin. It's a standalone package that depends on `kirin-toolchain` from PyPI, and it doesn't fork or patch Kirin.

## How it works

- **Export.** Kernels are written in MLIR's generic form. Every operation has the same shape there, so one emitter covers every Kirin dialect, including dialects defined in other packages.
- **Verification.** IRDL dialect definitions are generated straight from Kirin's statement declarations. `mlir-opt` loads them at run time and checks each operation's operand and result counts, its attributes, and that its types are well-formed Kirin types, with no C++ involved.
- **Import.** MLIR text, ours or `mlir-opt`'s printout, is parsed back into Kirin kernels. Those kernels run in Kirin's interpreter and give the same results as the originals.

Anything that can't be represented faithfully raises an error rather than being approximated.

## Prerequisites

- Python 3.10 or later.
- `mlir-opt`, from Homebrew's `llvm` on macOS or apt.llvm.org on Linux. It's a native binary, not a Python package, and MLIR's Python bindings aren't needed.

```bash
brew install llvm                               # keg-only, so:
export PATH="$(brew --prefix llvm)/bin:$PATH"
uv sync --all-groups --all-extras
```

## Usage

```python
from kirin_mlir import emit_mlir, import_mlir, prepare_for_mlir

text = emit_mlir(prepare_for_mlir(kernel))    # Kirin -> MLIR
module = import_mlir(text, kernel.dialects)   # MLIR -> Kirin
module.entry(3, 4)                            # runs in Kirin's interpreter
```

Generate IRDL for your dialects and verify a module:

```bash
uv run kirin-mlir irdl --all -o kirin.irdl.mlir
mlir-opt --irdl-file=kirin.irdl.mlir --allow-unregistered-dialect module.mlir
```

## bloqade.stim

The optional `stim` extra adds [bloqade-circuit](https://github.com/QuEraComputing/bloqade-circuit)'s stim dialects as a worked example. The test programs use all 53 stim statements, and each one verifies against the checked-in `kirin_stim.irdl.mlir`. After a round trip through MLIR, each program produces the same circuit as the original when checked with [stim](https://github.com/quantumlib/Stim).

## Limitations

- IRDL can't declare regions, branches or terminators, so control-flow dialects (`func`, `cf`, `scf`, stim's `REPEAT`) are parsed by MLIR but not verified.
- IRDL requires every operand in a variadic group to have the same type, so `mlir-opt` rejects some valid programs, such as the tuple `(x, 1.5)`.

## Tests

```bash
uv run pytest -q -rs
```

The tests use real `mlir-opt` as the reference. Without it on `PATH`, those tests skip rather than fail, so check the skip summary.
