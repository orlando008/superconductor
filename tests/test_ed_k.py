"""Momentum-resolved exact diagonalization (issue 6), checked against plain ED."""

import numpy as np
import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import ExactDiagonalization, MomentumED
from superconductor.solvers.ed import build_hamiltonian, sector_dimension
from superconductor.solvers.ed_k import OrbitBasis

ED, EDK = ExactDiagonalization(), MomentumED()


def model(lattice, nu, nd, **kw):
    params = {"t_prime": -0.3, "U": 4.5, "mu": 0.3} | kw
    return HubbardModel(lattice=lattice, filling=Filling(n_up=nu, n_dn=nd), **params)


# The lattices the issue asks for (2x2, 4x2, 3x3, 4x1), plus periodic 3x2.
CASES = [
    (SquareLattice(lx=2, ly=2), (2, 1)),
    (SquareLattice(lx=2, ly=2), (2, 2)),
    (SquareLattice(lx=4, ly=1), (2, 2)),
    (SquareLattice(lx=4, ly=1), (3, 1)),
    (SquareLattice(lx=4, ly=2), (2, 2)),
    (SquareLattice(lx=4, ly=2), (3, 2)),
    (SquareLattice(lx=3, ly=3), (2, 2)),
    (SquareLattice(lx=3, ly=3), (3, 2)),
    (SquareLattice(lx=3, ly=2), (2, 2)),
    (SquareLattice(lx=3, ly=2), (3, 3)),
]


def _block_spectra(m):
    basis = OrbitBasis(m)
    spectra, dims = [], 0
    for k in basis.momenta():
        ham, reps = basis.block(k)
        dims += ham.shape[0]
        if ham.shape[0]:
            assert abs(ham - ham.conj().T).max() < 1e-12, f"block {k} not Hermitian"
            spectra.extend(np.linalg.eigvalsh(ham.toarray()))
    return np.sort(spectra), dims


@pytest.mark.parametrize("lattice,filling", CASES)
def test_union_of_block_spectra_equals_full_spectrum(lattice, filling):
    """Every eigenvalue of the full sector appears in exactly one momentum block."""
    m = model(lattice, *filling)
    spectra, dims = _block_spectra(m)
    assert dims == sector_dimension(m)
    full = np.linalg.eigvalsh(build_hamiltonian(m)[0].toarray())
    assert np.allclose(spectra, full, atol=1e-9)


@pytest.mark.parametrize("lattice,filling", CASES)
@pytest.mark.parametrize("U", [0.0, 6.0])
def test_ground_state_matches_ed(lattice, filling, U):
    """Energy, double occupancy (averaged over the whole ground manifold, across all momenta)
    and the degeneracy must agree with ED."""
    m = model(lattice, *filling, U=U)
    k, e = EDK.solve(m), ED.solve(m)
    assert k["energy"] == pytest.approx(e["energy"], abs=1e-9)
    assert k["double_occupancy"] == pytest.approx(e["double_occupancy"], abs=1e-10)
    assert k.metadata["ground_state_degeneracy"] == e.metadata["ground_state_degeneracy"]
    assert k.metadata["max_residual"] < 1e-7


def test_degenerate_ground_state_across_momenta():
    """3x3, (3,2): four-fold degenerate, one state at each of four momenta (as ED finds)."""
    m = model(SquareLattice(lx=3, ly=3), 3, 2)
    res = EDK.solve(m)
    assert res.metadata["ground_state_degeneracy"] == 4
    assert len(res.metadata["ground_momenta"]) == 4


def test_blocks_are_real_exactly_when_all_phases_are_plus_or_minus_one():
    basis = OrbitBasis(model(SquareLattice(lx=4, ly=2), 3, 2))
    for k in basis.momenta():
        ham, _ = basis.block(k)
        zone_boundary = (2 * k[0]) % 4 == 0 and (2 * k[1]) % 2 == 0  # K components in {0, pi}
        assert (ham.dtype == np.float64) == zone_boundary, k


def test_workers_from_environment_and_not_recorded_as_a_setting(monkeypatch):
    monkeypatch.setenv("SUPERCONDUCTOR_WORKERS", "3")
    assert MomentumED().workers == 3
    assert "workers" not in MomentumED().settings()  # speed only, never the result


def test_parallel_workers_give_identical_results():
    m = model(SquareLattice(lx=3, ly=3), 3, 2)
    serial, parallel = MomentumED().solve(m), MomentumED(workers=2).solve(m)
    assert parallel["energy"] == pytest.approx(serial["energy"], abs=1e-12)
    assert parallel.metadata["ground_momenta"] == serial.metadata["ground_momenta"]


def test_fermion_signs_matter(monkeypatch):
    """Mutation test: dropping the translation signs breaks the spectrum check. The check is
    sensitive to exactly the subtle part of the construction."""
    original = OrbitBasis.__init__

    def without_signs(self, m):
        original(self, m)
        self.stab_sign[:] = 1
        self.state_sign[:] = 1
        self._build_hops(m)

    monkeypatch.setattr(OrbitBasis, "__init__", without_signs)
    m = model(SquareLattice(lx=4, ly=1), 2, 2)
    # Without the signs the blocks stop being Hermitian, or their spectra stop matching:
    # either way the checks above fail.
    try:
        spectra, dims = _block_spectra(m)
    except AssertionError as caught:
        assert "not Hermitian" in str(caught)
        return
    full = np.linalg.eigvalsh(build_hamiltonian(m)[0].toarray())
    assert dims != sector_dimension(m) or not np.allclose(spectra, full, atol=1e-6)


@pytest.mark.parametrize(
    "m,reason",
    [
        (model(SquareLattice(lx=3, ly=2, boundary_y="open"), 2, 2), "periodic"),
        (model(SquareLattice(lx=2, ly=2), 1, 1, temperature=0.5), "ground state"),
        (model(SquareLattice(lx=4, ly=4), 8, 8), "exceeds"),
    ],
)
def test_applicability(m, reason):
    assert reason in (EDK.applicability(m) or "")


def test_estimate_reports_block_size_and_less_memory_than_ed():
    m = model(SquareLattice(lx=4, ly=4), 5, 5)
    est, plain = EDK.estimate(m), ED.estimate(m)
    assert est.feasible and est.details["blocks"] == 16
    assert est.memory_bytes < plain.memory_bytes / 4
