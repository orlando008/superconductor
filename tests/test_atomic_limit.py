"""Atomic limit (t = t' = 0) at zero and finite temperature, canonical ensemble."""

from math import comb, log

import numpy as np
import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import AtomicLimit, ExactDiagonalization
from superconductor.solvers.ed import species_basis
from superconductor.solvers.exact_limits import double_occupancy_degeneracies

ATOMIC, ED = AtomicLimit(), ExactDiagonalization()


@pytest.mark.parametrize(
    "n,nu,nd",
    [(1, 1, 1), (2, 1, 1), (4, 2, 2), (4, 3, 2), (5, 3, 3), (6, 4, 5), (6, 0, 3), (7, 3, 2)],
)
def test_degeneracies_match_brute_force_enumeration(n, nu, nd):
    up, dn = species_basis(n, nu), species_basis(n, nd)
    doubles = np.bitwise_count(up[:, None] & dn[None, :]).ravel()
    brute = {int(d): int(c) for d, c in zip(*np.unique(doubles, return_counts=True), strict=True)}
    assert double_occupancy_degeneracies(n, nu, nd) == brute


@pytest.mark.parametrize("n,nu,nd", [(10, 5, 5), (16, 9, 7), (40, 22, 21)])
def test_degeneracies_sum_to_sector_dimension(n, nu, nd):
    """Vandermonde identity: sum_d g(d) = C(N, N_up) C(N, N_dn), also for large N."""
    assert sum(double_occupancy_degeneracies(n, nu, nd).values()) == comb(n, nu) * comb(n, nd)


LATTICES = [
    SquareLattice(lx=2, ly=2),
    SquareLattice(lx=4, ly=1),
    SquareLattice(lx=3, ly=2, boundary_x="open"),
]


@pytest.mark.parametrize("lattice", LATTICES)
@pytest.mark.parametrize("filling", [(2, 2), (3, 1), (3, 3), (4, 3)])  # incl. over half filling
@pytest.mark.parametrize("U", [-2.0, 0.0, 4.0, 10.0])
@pytest.mark.parametrize("T", [0.05, 0.5, 5.0])
def test_agrees_with_ed_at_finite_temperature(lattice, filling, U, T):
    nu, nd = filling
    if max(nu, nd) > lattice.num_sites:
        pytest.skip("filling exceeds lattice")
    m = HubbardModel(
        lattice=lattice, t=0.0, U=U, mu=0.3, temperature=T, filling=Filling(n_up=nu, n_dn=nd)
    )
    exact, ed = ATOMIC.solve(m), ED.solve(m)
    assert set(exact.observables) == set(ed.observables)
    for name in exact.observables:
        assert exact[name] == pytest.approx(ed[name], abs=1e-10), name
    assert exact["kinetic_energy"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("U", [-3.0, 0.0, 4.0])
def test_zero_temperature_agrees_with_ed_including_degenerate_cases(U):
    m = HubbardModel(lattice=SquareLattice(lx=3, ly=2), t=0.0, U=U, filling=Filling(n_up=3, n_dn=2))
    exact, ed = ATOMIC.solve(m), ED.solve(m)
    for name in ed.observables:
        assert exact[name] == pytest.approx(ed[name], abs=1e-12), name
    assert exact.metadata["ground_state_degeneracy"] == ed.metadata["ground_state_degeneracy"]


def test_infinite_temperature_limit():
    """T -> infinity: every state equally likely.

    Entropy -> ln[C(N, N_up) C(N, N_dn)], and <D> -> N_up N_dn / N (each site is occupied by
    an up electron with probability N_up/N and, independently, a down one with N_dn/N).
    """
    n, nu, nd = 6, 4, 3
    m = HubbardModel(
        lattice=SquareLattice(lx=3, ly=2),
        t=0.0,
        U=5.0,
        temperature=1e9,
        filling=Filling(n_up=nu, n_dn=nd),
    )
    res = ATOMIC.solve(m)
    assert res["entropy"] == pytest.approx(log(comb(n, nu) * comb(n, nd)), rel=1e-7)
    assert res["double_occupancy"] * n == pytest.approx(nu * nd / n, rel=1e-7)


def test_low_temperature_approaches_ground_state():
    base = dict(lattice=SquareLattice(lx=2, ly=2), t=0.0, U=4.0, filling=Filling(n_up=3, n_dn=2))
    cold = ATOMIC.solve(HubbardModel(temperature=1e-3, **base))
    ground = ATOMIC.solve(HubbardModel(**base))
    assert cold["energy"] == pytest.approx(ground["energy"], abs=1e-12)
    assert cold["double_occupancy"] == pytest.approx(ground["double_occupancy"], abs=1e-12)
    # Residual entropy is ln(ground-state degeneracy).
    assert cold["entropy"] == pytest.approx(
        log(ground.metadata["ground_state_degeneracy"]), abs=1e-9
    )


def test_large_lattice_is_numerically_stable():
    """No overflow for sizes far beyond ED (g(d) ~ 1e30 here)."""
    m = HubbardModel(
        lattice=SquareLattice(lx=10, ly=10),
        t=0.0,
        U=8.0,
        temperature=0.7,
        filling=Filling(n_up=45, n_dn=45),
    )
    res = ATOMIC.solve(m)
    assert all(np.isfinite(o.value) for o in res.observables.values())
    assert 0.0 <= res["double_occupancy"] <= 0.45
