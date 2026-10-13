"""The stim sample programs of issue 05, checked before anything exports them.

Later steps export these programs to MLIR and compare the circuit recovered
from it with :func:`stim_kernels.reference`. That comparison only means
something if the reference is a circuit stim can analyse: every detector and
observable deterministic, and the noise present. Skipped without the ``stim``
extra (``uv sync --all-groups --all-extras``).
"""

import pytest

pytest.importorskip("bloqade.stim", reason="needs the stim extra")

from stim_kernels import EVERY, reference, repetition


@pytest.mark.parametrize("kernel", EVERY, ids=lambda kernel: kernel.sym_name)
def test_bloqade_turns_each_program_into_a_circuit_stim_can_analyse(kernel):
    circuit = reference(kernel)
    # Raises unless every detector and observable is deterministic.
    model = circuit.detector_error_model()
    assert model.num_detectors == circuit.num_detectors > 0


def test_the_noise_reaches_the_error_model():
    """An export that dropped a noise channel must be detectable later."""
    assert reference(repetition).detector_error_model().num_errors > 0


def test_the_programs_use_four_of_the_five_stim_dialects():
    """``stim.cf`` is missing: ``REPEAT`` cannot be written under ``@stim.main``."""
    used = {
        stmt.dialect.name
        for kernel in EVERY
        for stmt in kernel.callable_region.walk()
        if stmt.dialect is not None and stmt.dialect.name.startswith("stim.")
    }
    assert used == {"stim.aux", "stim.collapse", "stim.gate", "stim.noise"}
