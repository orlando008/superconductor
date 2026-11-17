"""Bethe ansatz (Lieb-Wu) solver for periodic 1D Hubbard rings (issue 4)."""

import numpy as np
import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import BetheAnsatz, ExactDiagonalization
from superconductor.solvers.bethe import ground_state, half_filled_closed_shell

BETHE, ED = BetheAnsatz(), ExactDiagonalization()


def ring(L, nu, nd, U, **kw):
    return HubbardModel(
        lattice=SquareLattice(lx=L, ly=1), U=U, filling=Filling(n_up=nu, n_dn=nd), **kw
    )


def test_two_site_ring_matches_closed_form():
    """Periodic L = 2 counts the bond twice (effective hopping 2t), so
    E = (U - sqrt(U^2 + 16 (2t)^2)) / 2, the two-site formula with t -> 2t."""
    for U in (0.5, 4.0, 20.0):
        res = BETHE.solve(ring(2, 1, 1, U))
        assert res["energy"] == pytest.approx((U - np.sqrt(U**2 + 64)) / 2, abs=1e-12)


# Each case is labelled with the structural feature it exercises (see the module docstring).
CASES = [
    (4, 1, 1, "closed shell"),
    (4, 2, 1, "open shell, needs shifted quantum numbers"),
    (4, 2, 2, "half filling, L = 4n: open shell"),
    (6, 2, 1, "higher-spin ground state at strong U"),
    (6, 2, 2, "S = 1 ground state (finite-size Hund's rule)"),
    (6, 3, 3, "half filling, L = 4n + 2: closed-shell shortcut"),
    (6, 3, 1, "spin-polarized sector"),
    (7, 2, 2, "odd ring"),
    (8, 2, 2, "low density"),
    (8, 3, 2, "open shell, odd N"),
]


@pytest.mark.parametrize("L,nu,nd,feature", CASES)
@pytest.mark.parametrize("U", [1.0, 4.0, 10.0])
def test_agrees_with_ed(L, nu, nd, feature, U):
    m = ring(L, nu, nd, U, mu=0.15)
    b, e = BETHE.solve(m), ED.solve(m)
    assert b["energy"] == pytest.approx(e["energy"], abs=1e-9), feature
    d = b.observables["double_occupancy"]
    assert d.error < 1e-6
    assert abs(d.value - e["double_occupancy"]) <= max(2 * d.error, 1e-10), feature


def test_fully_polarized_ground_state_found_in_lower_spin_sector():
    """6 sites, 3 electrons (2 up, 1 down), U = 12: the ground state has S = 3/2 and is a free
    spinless Fermi sea (levels -2, -1, -1, so E = -4 exactly), with no double occupancy."""
    res = BETHE.solve(ring(6, 2, 1, 12.0))
    assert res.metadata["total_spin"] == 1.5
    assert res["energy"] == pytest.approx(-4.0, abs=1e-12)
    assert res["double_occupancy"] == pytest.approx(0.0, abs=1e-9)


def test_quantum_numbers_follow_the_parity_rules():
    """I_j in Z + M/2 and J_a in Z + (N - M + 1)/2, for the M of the chosen multiplet."""
    for L, nu, nd, _ in CASES:
        res = BETHE.solve(ring(L, nu, nd, 4.0))
        n = nu + nd
        mp = int(round(n / 2 - res.metadata["total_spin"]))
        for v in res.metadata["quantum_numbers_I"]:
            assert (v - mp / 2) % 1 == pytest.approx(0, abs=1e-12)
        for v in res.metadata["quantum_numbers_J"]:
            assert (v - (n - mp + 1) / 2) % 1 == pytest.approx(0, abs=1e-12)
        assert len(res.metadata["quantum_numbers_J"]) == mp


@pytest.mark.parametrize("U", [0.3, 4.0, 15.0])
def test_closed_shell_shortcut_equals_full_search(U):
    """The shortcut skips the search. On L = 6 it must find what the full search finds.
    (L = 10 is checked against ED in the cross-check suite instead: the full search there
    takes tens of seconds, too slow for every CI run.)"""
    L = 6
    assert half_filled_closed_shell(L, L, L // 2)
    fast = ground_state(L, L, L // 2, U, search=False)
    full = ground_state(L, L, L // 2, U, search=True)
    assert fast[0] == pytest.approx(full[0], abs=1e-10)


def test_beyond_ed_reach_is_fast_and_sensible():
    """L = 22 half filling (ED would need about 5e11 states): the shortcut solves it quickly,
    and the energy per site lies between the U = 0 value and the infinite-chain value at U = 4
    (Lieb-Wu: e(U=4) = -0.5737...; finite periodic rings with L = 4n + 2 lie below it)."""
    res = BETHE.solve(ring(22, 11, 11, 4.0))
    e = res["energy_per_site"]
    free = -4 / (22 * np.sin(np.pi / 22))  # free ring at half filling, L = 4n + 2
    assert free < e < -0.5737


@pytest.mark.parametrize(
    "m,reason",
    [
        (
            HubbardModel(lattice=SquareLattice(lx=2, ly=2), U=4, filling=Filling(n_up=1, n_dn=1)),
            "1D",
        ),
        (ring(4, 1, 1, 4.0).with_updates(t_prime=-0.2), "t' = 0"),
        (ring(4, 1, 1, 0.0), "U > 0"),
        (ring(4, 1, 1, 4.0, temperature=0.5), "ground state"),
        (ring(4, 3, 2, 4.0), "half filling"),
        (
            HubbardModel(
                lattice=SquareLattice(lx=4, ly=1, boundary_x="open"),
                U=4,
                filling=Filling(n_up=1, n_dn=1),
            ),
            "periodic",
        ),
    ],
)
def test_applicability_reasons(m, reason):
    assert reason in (BETHE.applicability(m) or "")
