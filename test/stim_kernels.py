"""Sample ``bloqade.stim`` programs, the payload of issue 05.

Small circuits a bloqade author would write, each chosen to exercise something
the export has to get right:

- :func:`bell`: a reset, a gate, ``CX``, a measurement and a detector. The
  smallest program that touches gates, measurements and records.
- :func:`repetition`: a distance-3 repetition code with noise. ``CX`` with
  several control/target pairs (two variadic groups, so ``operandSegmentSizes``),
  two noise channels, measurement flip probabilities, detectors and an
  observable.
- :func:`pauli_products`: qubit coordinates, ``TICK``, and Pauli products
  measured with ``MPP`` and applied with ``SPP``. ``stim.pauli_string`` is the
  one stim statement whose result is a stim-specific type.

``@stim.main`` has no ``py.assign``, so a value cannot be named; each Pauli
string is written where it is used.

None of them uses ``stim.cf``. Its one statement, ``REPEAT``, cannot be written
under ``@stim.main``: bloqade only creates it when ``SquinToStimPass`` rewrites
an ``scf.for`` in a squin kernel (``bloqade/stim/rewrite/scf_for_to_repeat.py``).
"""

from __future__ import annotations

import stim as stim_sim
from kirin import ir
from bloqade import stim


@stim.main
def bell():
    stim.rz(targets=(0, 1))
    stim.h(targets=(0,))
    stim.cx(controls=(0,), targets=(1,))
    stim.mz(p=0.0, targets=(0, 1))
    stim.detector(coord=(0, 0), targets=(stim.rec(-1), stim.rec(-2)))


@stim.main
def repetition():
    stim.rz(targets=(0, 1, 2, 3, 4))
    stim.x_error(p=0.01, targets=(0, 2, 4))
    stim.cx(controls=(0, 2), targets=(1, 3))
    stim.cx(controls=(2, 4), targets=(1, 3))
    stim.depolarize2(p=0.001, targets=(0, 1, 2, 3))
    stim.mz(p=0.02, targets=(1, 3))
    stim.detector(coord=(1, 0), targets=(stim.rec(-2),))
    stim.detector(coord=(3, 0), targets=(stim.rec(-1),))
    stim.mz(p=0.0, targets=(0, 2, 4))
    stim.observable_include(idx=0, targets=(stim.rec(-1),))


@stim.main
def pauli_products():
    stim.qubit_coords(coord=(0.0, 0.0), target=0)
    stim.qubit_coords(coord=(1.0, 0.0), target=1)
    stim.rx(targets=(0, 1))
    stim.tick()
    stim.spp(
        targets=(
            stim.pauli_string(
                string=("X", "X"), flipped=(False, False), targets=(0, 1)
            ),
        )
    )
    stim.mpp(
        p=0.0,
        targets=(
            stim.pauli_string(
                string=("X", "X"), flipped=(False, False), targets=(0, 1)
            ),
        ),
    )
    stim.detector(coord=(0, 1), targets=(stim.rec(-1),))


EVERY = [bell, repetition, pauli_products]


def reference(kernel: ir.Method) -> stim_sim.Circuit:
    """The circuit bloqade's own ``emit.stim`` produces for ``kernel``.

    What the circuit recovered from the exported MLIR is compared with.
    """
    return stim.Circuit(kernel)
