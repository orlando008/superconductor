"""Weak- and strong-coupling limits as checks on ED (issue 5)."""

from functools import reduce
from itertools import combinations

import numpy as np
import pytest

import superconductor.solvers.ed as ed_module
from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.validation import (
    heisenberg_ground_energy,
    run_perturbative_checks,
    strong_coupling_check,
    weak_coupling_check,
)
from superconductor.validation.perturbative import heisenberg_couplings

DIMER = SquareLattice(lx=2, ly=1, boundary_x="open")


def model(lattice, nu, nd, **kw):
    return HubbardModel(lattice=lattice, filling=Filling(n_up=nu, n_dn=nd), **({"U": 1.0} | kw))


# ---- the reference Heisenberg solver --------------------------------------------------------


def test_heisenberg_two_site_singlet():
    """Textbook: J S1.S2 has singlet energy -3J/4 (and triplet +J/4)."""
    assert heisenberg_ground_energy({(0, 1): 1.0}, 2, 1) == pytest.approx(-0.75, abs=1e-14)
    assert heisenberg_ground_energy({(0, 1): 1.0}, 2, 2) == pytest.approx(0.25, abs=1e-14)


def test_heisenberg_four_site_ring():
    """Textbook: the 4-site ring J sum S_i.S_{i+1} has ground energy -2J."""
    ring = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (0, 3): 1.0}
    assert heisenberg_ground_energy(ring, 4, 2) == pytest.approx(-2.0, abs=1e-13)


def _brute_force_heisenberg(couplings, n):
    """Independent construction: full 2^N space from Pauli matrices, then the lowest level."""
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2

    def op(single, site):
        mats = [np.eye(2)] * n
        mats[site] = single
        return reduce(np.kron, mats)

    ham = sum(c * sum(op(s, i) @ op(s, j) for s in (sx, sy, sz)) for (i, j), c in couplings.items())
    total_sz = sum(op(sz, i) for i in range(n))
    return np.real(ham), np.real(np.diag(total_sz))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heisenberg_matches_brute_force_in_every_sector(seed):
    rng = np.random.default_rng(seed)
    n = 5
    couplings = {pair: float(rng.uniform(-1, 2)) for pair in combinations(range(n), 2)}
    ham, sz = _brute_force_heisenberg(couplings, n)
    for n_up in range(n + 1):
        # Bit 1 = up in our solver; in the Kronecker basis spin up is index 0.
        keep = np.isclose(sz, n_up - n / 2)
        expected = np.linalg.eigvalsh(ham[np.ix_(keep, keep)])[0]
        assert heisenberg_ground_energy(couplings, n, n_up) == pytest.approx(expected, abs=1e-12)


def test_couplings_include_doubled_bonds_and_diagonal_hopping():
    """J_ij = 4 h_ij^2 / U uses the total hopping between i and j (convention in lattice.py)."""
    m = model(SquareLattice(lx=2, ly=2), 2, 2, t=1.0, t_prime=-0.3)
    j = heisenberg_couplings(m, U=8.0)
    assert j[(0, 1)] == pytest.approx(4 * 2.0**2 / 8)  # periodic length 2: bond counted twice
    assert j[(0, 3)] == pytest.approx(4 * (4 * 0.3) ** 2 / 8)  # four diagonal images


# ---- the checks against analytic results -------------------------------------------------------


def test_dimer_weak_coupling_coefficient():
    """Exact dimer: E = (U - sqrt(U^2 + 16 t^2))/2 = -2t + U/2 - U^2/(16 t) + O(U^4)."""
    out = weak_coupling_check(model(DIMER, 1, 1, t=1.0))
    assert out.status == "pass"
    u = 0.005
    assert out.remainders[0] / u**2 == pytest.approx(-1 / 16, rel=1e-3)


def test_dimer_strong_coupling_coefficient():
    """Same exact formula at large U: E = -4t^2/U + 16 t^4/U^3 + O(U^-5); Heisenberg gives -J."""
    out = strong_coupling_check(model(DIMER, 1, 1, t=1.0))
    assert out.status == "pass"
    u = 200.0
    assert out.remainders[-1] * u**3 == pytest.approx(16.0, rel=1e-3)


def test_weak_coupling_remainders_are_never_positive():
    """Variational bound: E0(U) <= E_free + U D_free for every U >= 0."""
    for o in run_perturbative_checks():
        if o.name == "weak_coupling" and o.status != "skipped":
            assert all(r <= 1e-10 for r in o.remainders), o


def test_whole_suite_passes_with_both_checks_exercised():
    outcomes = run_perturbative_checks()
    assert not [o for o in outcomes if o.status == "fail"]
    passed = {o.name for o in outcomes if o.status == "pass"}
    assert passed == {"weak_coupling", "strong_coupling"}


@pytest.mark.parametrize(
    "check,m,reason",
    [
        (weak_coupling_check, model(SquareLattice(lx=2, ly=2), 2, 2), "open shell"),
        (weak_coupling_check, model(SquareLattice(lx=2, ly=2), 2, 0), "empty"),
        (weak_coupling_check, model(DIMER, 1, 1, temperature=1.0), "T > 0"),
        (strong_coupling_check, model(SquareLattice(lx=2, ly=2), 1, 1), "half filling"),
        (strong_coupling_check, model(DIMER, 1, 1, t=0.0), "no hopping"),
    ],
)
def test_inapplicable_cases_are_skipped_with_a_reason(check, m, reason):
    out = check(m)
    assert out.status == "skipped" and reason in out.detail


# ---- mutation tests: errors invisible at U = 0 and t = 0 are caught ----------------------------

_original_build = ed_module.build_hamiltonian


def _interaction_off_by_a_tenth_percent(model):
    """A plausible bug: the interaction enters 0.1% too strong."""
    return _original_build(model.with_updates(U=model.U * 1.001))


def test_slightly_wrong_interaction_is_caught_by_both_limits(monkeypatch):
    monkeypatch.setattr(ed_module, "build_hamiltonian", _interaction_off_by_a_tenth_percent)
    # The exact-limit cross-checks can't see this bug: at U = 0 or t = 0 nothing changes...
    from superconductor.validation import crosscheck

    free_model = model(SquareLattice(lx=3, ly=2), 3, 3, U=0.0)
    assert crosscheck(free_model).passed
    # ...but the perturbative checks fail on both sides.
    weak = weak_coupling_check(model(SquareLattice(lx=3, ly=2), 3, 3))
    strong = strong_coupling_check(model(SquareLattice(lx=3, ly=2), 3, 3))
    assert weak.status == "fail" and strong.status == "fail"
    assert "variational bound" in weak.detail
    # The bug adds a 1/U term to the remainder, so the fitted slope moves well away from -3
    # (to about -2.5 over this range of U, where the 1/U and 1/U^3 terms mix).
    from superconductor.validation.perturbative import EXPONENT_TOLERANCE, STRONG_EXPONENT

    assert abs(strong.fitted_exponent - STRONG_EXPONENT) > 2 * EXPONENT_TOLERANCE
