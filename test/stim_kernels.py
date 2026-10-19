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
- :func:`clifford_gates`: every other Clifford gate, each undone by its
  ``dagger`` where it is not its own inverse, so the measurements stay
  deterministic. ``T`` has no wrapper and is called as its statement class.
- :func:`pauli_measurements`: resets and measurements in every Pauli basis,
  single-qubit and pairwise, each measuring a state it has an eigenvalue on.
- :func:`noise_channels`: every noise channel, including the five that stim
  does not know and bloqade emits as tagged ``I_ERROR``.
- :func:`rotations`: ``R_X``, ``R_Y``, ``R_Z``, ``U3`` and ``REPEAT``, which
  cannot be written under ``@stim.main`` (no wrapper, and their statements
  cannot be called from Python). bloqade only creates them when
  ``SquinToStimPass`` lowers a squin kernel; an ``scf.for`` over a constant
  ``range`` whose loop variable is unused becomes ``REPEAT``.

``@stim.main`` has no ``py.assign``, so a value cannot be named; each Pauli
string is written where it is used.

Together they use every statement of stim's five dialects, which
``test_stim_kernels.py`` pins.
"""

from __future__ import annotations

import stim as stim_sim
from kirin import ir
from bloqade import stim, squin
from bloqade.stim.passes import SquinToStimPass
from bloqade.stim.dialects import gate, noise

from kirin_mlir import find_dialects


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


@stim.main
def clifford_gates():
    stim.rz(targets=(0, 1))
    stim.identity(targets=(0, 1))
    stim.x(targets=(0,))
    stim.y(targets=(1,))
    stim.z(targets=(0,))
    stim.s(targets=(0,))
    stim.s(targets=(0,), dagger=True)
    stim.sqrt_x(targets=(1,))
    stim.sqrt_x(targets=(1,), dagger=True)
    stim.sqrt_y(targets=(0,))
    stim.sqrt_y(targets=(0,), dagger=True)
    stim.sqrt_z(targets=(1,))
    stim.sqrt_z(targets=(1,), dagger=True)
    gate.T(targets=(0,))
    gate.T(targets=(0,), dagger=True)
    stim.cz(controls=(0,), targets=(1,))
    stim.cy(controls=(0,), targets=(1,))
    stim.swap(targets=(0, 1))
    stim.mz(p=0.0, targets=(0, 1))
    stim.detector(coord=(0, 0), targets=(stim.rec(-2),))
    stim.detector(coord=(1, 0), targets=(stim.rec(-1),))


@stim.main
def pauli_measurements():
    stim.rx(targets=(0,))
    stim.ry(targets=(1,))
    stim.mx(p=0.0, targets=(0,))
    stim.my(p=0.0, targets=(1,))
    stim.rz(targets=(0, 1))
    stim.mzz(p=0.0, targets=(0, 1))
    stim.rx(targets=(0, 1))
    stim.mxx(p=0.0, targets=(0, 1))
    stim.ry(targets=(0, 1))
    stim.myy(p=0.01, targets=(0, 1))
    stim.detector(coord=(0, 0), targets=(stim.rec(-5),))
    stim.detector(coord=(1, 0), targets=(stim.rec(-4),))
    stim.detector(coord=(0, 1), targets=(stim.rec(-3),))
    stim.detector(coord=(0, 2), targets=(stim.rec(-2),))
    stim.detector(coord=(0, 3), targets=(stim.rec(-1),))


@stim.main
def noise_channels():
    stim.rz(targets=(0, 1))
    stim.depolarize1(p=0.01, targets=(0, 1))
    stim.pauli_channel1(px=0.01, py=0.02, pz=0.03, targets=(0,))
    stim.pauli_channel2(
        pix=0.001,
        piy=0.001,
        piz=0.001,
        pxi=0.001,
        pxx=0.001,
        pxy=0.001,
        pxz=0.001,
        pyi=0.001,
        pyx=0.001,
        pyy=0.001,
        pyz=0.001,
        pzi=0.001,
        pzx=0.001,
        pzy=0.001,
        pzz=0.001,
        targets=(0, 1),
    )
    stim.y_error(p=0.01, targets=(1,))
    stim.z_error(p=0.01, targets=(0,))
    noise.TrivialError(probs=(0.01,), targets=(0,))
    noise.TrivialCorrelatedError(probs=(0.01,), targets=(0, 1))
    stim.qubit_loss(probs=(0.01,), targets=(0,))
    stim.correlated_qubit_loss(probs=(0.01,), targets=(0, 1))
    noise.QubitLeakage(probs=(0.01,), targets=(1,))
    stim.mz(p=0.0, targets=(0, 1))
    stim.detector(coord=(0, 0), targets=(stim.rec(-2),))
    stim.detector(coord=(1, 0), targets=(stim.rec(-1),))


def from_squin(kernel: ir.Method) -> ir.Method:
    """A copy of ``kernel``, lowered to a stim program by bloqade's own pass."""
    lowered = kernel.similar()
    SquinToStimPass(lowered.dialects)(lowered)
    return lowered.similar(stim.main)


@squin.kernel
def rotations():
    q = squin.qalloc(2)
    squin.rx(0.1, q[0])
    squin.ry(0.2, q[0])
    squin.rz(0.3, q[0])
    squin.u3(0.1, 0.2, 0.3, q[1])
    for _ in range(3):
        squin.x(q[0])
        squin.x(q[0])
    m = squin.broadcast.measure(q)
    squin.set_detector([m[0]], coordinates=[0, 0])


rotations = from_squin(rotations)

EVERY = [
    bell,
    repetition,
    pauli_products,
    clifford_gates,
    pauli_measurements,
    noise_channels,
    rotations,
]


def dialects() -> list[ir.Dialect]:
    """Stim's five dialects, as the installed bloqade declares them."""
    found = find_dialects(["bloqade.stim"])
    return [d for name, d in found.items() if name.startswith("stim.")]


def reference(kernel: ir.Method) -> stim_sim.Circuit:
    """The circuit bloqade's own ``emit.stim`` produces for ``kernel``.

    What the circuit recovered from the exported MLIR is compared with.
    """
    return stim.Circuit(kernel)
