"""Free fermions (U = 0) at finite temperature, canonical ensemble."""

from itertools import combinations
from math import comb, exp, log

import numpy as np
import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import ExactDiagonalization, FreeFermions
from superconductor.solvers.exact_limits import canonical_occupations, log_elementary_symmetric

FREE, ED = FreeFermions(), ExactDiagonalization()


def brute_force(eps, n, beta):
    """Expected values from first principles: enumerate every n-fermion state (subset of levels)."""
    states = list(combinations(range(len(eps)), n))
    weights = np.array([exp(-beta * sum(eps[j] for j in s)) for s in states])
    z = weights.sum()
    occ = np.zeros(len(eps))
    for w, s in zip(weights, states, strict=True):
        occ[list(s)] += w / z
    return occ, log(z)


def test_elementary_symmetric_polynomials_match_direct_sums():
    x = np.array([0.3, 1.7, 2.2, 0.05, 1.0])
    logs = log_elementary_symmetric(np.log(x), len(x))
    for k in range(len(x) + 1):
        direct = sum(np.prod([x[j] for j in s]) for s in combinations(range(len(x)), k))
        assert np.exp(logs[k]) == pytest.approx(direct, rel=1e-12), k


@pytest.mark.parametrize(
    "eps",
    [
        np.array([-2.0, -0.5, 0.0, 0.0, 1.3, 2.0]),  # includes a degenerate pair
        np.array([-4.0, 0.0, 0.0, 4.0]),  # periodic 2x2 levels
    ],
)
@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("beta", [0.2, 1.0, 5.0])
def test_occupations_match_brute_force_enumeration(eps, n, beta):
    occ, log_z = canonical_occupations(eps, n, beta)
    exp_occ, exp_log_z = brute_force(eps, n, beta)
    assert log_z == pytest.approx(exp_log_z, abs=1e-12)
    assert np.allclose(occ, exp_occ, atol=1e-12)
    assert occ.sum() == pytest.approx(n, abs=1e-12)


LATTICES = [
    SquareLattice(lx=2, ly=2),
    SquareLattice(lx=4, ly=1),
    SquareLattice(lx=3, ly=2, boundary_x="open"),
]


@pytest.mark.parametrize("lattice", LATTICES)
@pytest.mark.parametrize("filling", [(1, 1), (2, 1), (2, 2), (3, 2), (4, 3)])
@pytest.mark.parametrize("T", [0.05, 0.5, 5.0])
def test_agrees_with_ed(lattice, filling, T):
    nu, nd = filling
    if max(nu, nd) > lattice.num_sites:
        pytest.skip("filling exceeds lattice")
    m = HubbardModel(
        lattice=lattice,
        t=1.0,
        t_prime=-0.3,
        U=0.0,
        mu=0.2,
        temperature=T,
        filling=Filling(n_up=nu, n_dn=nd),
    )
    free, ed = FREE.solve(m), ED.solve(m)
    assert set(free.observables) == set(ed.observables)
    for name in ed.observables:
        assert free[name] == pytest.approx(ed[name], abs=1e-10), name


@pytest.mark.parametrize(
    "lattice,filling",
    [
        (SquareLattice(lx=2, ly=2), (2, 2)),  # open shell: levels -4, 0, 0, 4
        (SquareLattice(lx=4, ly=1), (2, 1)),  # open shell for the up species
        (SquareLattice(lx=3, ly=2), (3, 3)),
    ],
)
def test_low_temperature_reproduces_ground_state_including_open_shells(lattice, filling):
    base = dict(lattice=lattice, U=0.0, filling=Filling(n_up=filling[0], n_dn=filling[1]))
    cold = FREE.solve(HubbardModel(temperature=1e-3, **base))
    ground = FREE.solve(HubbardModel(**base))
    for name in ground.observables:
        assert cold[name] == pytest.approx(ground[name], abs=1e-9), name


def test_infinite_temperature_limit():
    """T -> infinity: every state equally likely.

    Entropy -> ln[C(N, N_up) C(N, N_dn)], and every site is occupied with probability N_s/N
    independently for each species, so N * D -> N_up N_dn / N.
    """
    n, nu, nd = 6, 4, 3
    m = HubbardModel(
        lattice=SquareLattice(lx=3, ly=2),
        t_prime=-0.3,
        U=0.0,
        temperature=1e9,
        filling=Filling(n_up=nu, n_dn=nd),
    )
    res = FREE.solve(m)
    assert res["entropy"] == pytest.approx(log(comb(n, nu) * comb(n, nd)), rel=1e-7)
    assert res["double_occupancy"] * n == pytest.approx(nu * nd / n, rel=1e-7)


def test_large_lattice_low_temperature_is_numerically_stable():
    """10x10 at T = 0.01: far beyond ED, and where the alternating recursion fails."""
    m = HubbardModel(
        lattice=SquareLattice(lx=10, ly=10),
        t_prime=-0.3,
        U=0.0,
        temperature=0.01,
        filling=Filling(n_up=43, n_dn=41),
    )
    res = FREE.solve(m)
    assert all(np.isfinite(o.value) for o in res.observables.values())
    assert res["entropy"] >= -1e-9
    ground = FREE.solve(m.with_updates(temperature=0.0))
    assert res["energy"] >= ground["energy"] - 1e-9  # thermal energy can't be below E0
