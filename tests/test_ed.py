"""Exact diagonalization against analytically known answers and exact symmetries."""

import numpy as np
import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import ExactDiagonalization

ED = ExactDiagonalization()
DIMER = SquareLattice(lx=2, ly=1, boundary_x="open")


@pytest.mark.parametrize("U", [0.0, 0.5, 4.0, 20.0])
def test_two_site_hubbard_ground_state(U):
    t = 1.0
    r = np.sqrt(U**2 + 16 * t**2)
    res = ED.solve(HubbardModel(lattice=DIMER, t=t, U=U, filling=Filling(n_up=1, n_dn=1)))
    assert res["energy"] == pytest.approx((U - r) / 2, abs=1e-12)
    # Hellmann-Feynman: D_total = dE0/dU; per site divide by 2
    assert res["double_occupancy"] == pytest.approx((0.5 - U / (2 * r)) / 2, abs=1e-12)


@pytest.mark.parametrize("T", [0.1, 1.0, 5.0])
def test_two_site_hubbard_thermodynamics(T):
    U, t = 4.0, 1.0
    r = np.sqrt(U**2 + 16 * t**2)
    spectrum = np.array([(U - r) / 2, 0.0, U, (U + r) / 2])  # (1 up, 1 dn) sector
    p = np.exp(-spectrum / T) / np.exp(-spectrum / T).sum()
    res = ED.solve(
        HubbardModel(lattice=DIMER, t=t, U=U, temperature=T, filling=Filling(n_up=1, n_dn=1))
    )
    assert res["energy"] == pytest.approx(p @ spectrum, abs=1e-12)
    assert res["free_energy"] == pytest.approx(-T * np.log(np.exp(-spectrum / T).sum()), abs=1e-12)


def test_2x2_free_half_filling():
    # Periodic 2x2 single-particle levels: -4t, 0, 0, +4t
    res = ED.solve(
        HubbardModel(lattice=SquareLattice(lx=2, ly=2), U=0.0, filling=Filling(n_up=2, n_dn=2))
    )
    assert res["energy"] == pytest.approx(-8.0, abs=1e-12)
    assert res.metadata["ground_state_degeneracy"] == 4


@pytest.mark.parametrize(
    "lattice",
    [
        SquareLattice(lx=2, ly=2),
        SquareLattice(lx=4, ly=1),
        SquareLattice(lx=3, ly=2, boundary_x="open"),
    ],
)
@pytest.mark.parametrize("nu,nd", [(1, 1), (2, 1), (1, 0)])
def test_particle_hole_symmetry_on_bipartite_lattice(lattice, nu, nd):
    """E(nu, nd) = E(N-nu, N-nd) + U (nu + nd - N) when t' = 0 and mu = 0.

    This exercises fermion sign conventions: a wrong hopping sign breaks it.
    """
    n, U = lattice.num_sites, 3.0
    e = ED.solve(HubbardModel(lattice=lattice, U=U, filling=Filling(n_up=nu, n_dn=nd)))
    e_ph = ED.solve(HubbardModel(lattice=lattice, U=U, filling=Filling(n_up=n - nu, n_dn=n - nd)))
    assert e["energy"] == pytest.approx(e_ph["energy"] + U * (nu + nd - n), abs=1e-10)


def test_spin_flip_symmetry():
    lat = SquareLattice(lx=3, ly=2)
    a = ED.solve(HubbardModel(lattice=lat, t_prime=-0.3, U=5.0, filling=Filling(n_up=3, n_dn=2)))
    b = ED.solve(HubbardModel(lattice=lat, t_prime=-0.3, U=5.0, filling=Filling(n_up=2, n_dn=3)))
    assert a["energy"] == pytest.approx(b["energy"], abs=1e-10)


def test_chemical_potential_is_a_constant_shift_in_canonical_ensemble():
    lat = SquareLattice(lx=2, ly=2)
    f = Filling(n_up=2, n_dn=1)
    a = ED.solve(HubbardModel(lattice=lat, U=4.0, filling=f))
    b = ED.solve(HubbardModel(lattice=lat, U=4.0, mu=0.7, filling=f))
    assert b["energy"] == pytest.approx(a["energy"] - 0.7 * 3, abs=1e-12)
    assert b["kinetic_energy"] == pytest.approx(a["kinetic_energy"], abs=1e-12)


@pytest.mark.parametrize(
    "U,tp,filling",
    [(8.0, -0.3, (4, 4)), (4.0, -0.3, (3, 3)), (0.0, -0.3, (4, 4)), (6.0, 0.0, (4, 4))],
)
def test_lanczos_path_matches_dense(U, tp, filling):
    """The sparse path (incl. degenerate manifolds) must reproduce dense ED exactly."""
    m = HubbardModel(
        lattice=SquareLattice(lx=4, ly=2),
        t_prime=tp,
        U=U,
        filling=Filling(n_up=filling[0], n_dn=filling[1]),
    )
    dense = ExactDiagonalization(dense_limit=10**6).solve(m)
    sparse = ExactDiagonalization(dense_limit=0).solve(m)
    assert sparse.metadata["method"] == "lanczos+deflation"
    assert sparse.metadata["ground_state_degeneracy"] == dense.metadata["ground_state_degeneracy"]
    for name in dense.observables:
        assert sparse[name] == pytest.approx(dense[name], abs=1e-8), name


def test_diagonal_hamiltonian_with_huge_degeneracy():
    # Regression: ARPACK misses the E = 0 manifold of this t = 0 model entirely.
    m = HubbardModel(
        lattice=SquareLattice(lx=4, ly=2), t=0.0, U=6.0, filling=Filling(n_up=4, n_dn=4)
    )
    res = ExactDiagonalization(dense_limit=0).solve(m)
    assert res["energy"] == 0.0
    assert res.metadata["ground_state_degeneracy"] == 70
