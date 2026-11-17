from dataclasses import replace

import pytest

from superconductor import Filling, HubbardModel, Observable, SquareLattice
from superconductor.solvers import ExactDiagonalization, FreeFermions
from superconductor.validation import crosscheck, run_suite, sanity_check, standard_suite


def test_standard_suite_all_methods_agree():
    reports = run_suite()
    assert len(reports) == len(standard_suite())
    for r in reports:
        assert len(r.results) >= 2, f"{r.model.name}: fewer than two applicable solvers"
        assert r.passed, f"{r.model.name}: {r.failures}"


class SubtlyBrokenFree(FreeFermions):
    """A solver with a tiny bug, the kind that passes casual inspection."""

    name = "broken-free"

    def _solve(self, model):
        obs, meta = super()._solve(model)
        obs["energy"] = Observable(obs["energy"].value + 1e-6)
        return obs, meta


def test_crosscheck_flags_a_subtle_bug():
    m = HubbardModel(lattice=SquareLattice(lx=2, ly=2), filling=Filling(n_up=2, n_dn=1))
    report = crosscheck(m, solvers=[ExactDiagonalization(), SubtlyBrokenFree()])
    assert not report.passed
    assert [c.observable for c in report.failures] == ["energy"]


def test_crosscheck_records_why_solvers_were_skipped():
    m = HubbardModel(lattice=SquareLattice(lx=2, ly=2), U=4.0, filling=Filling(n_up=2, n_dn=2))
    report = crosscheck(m)
    assert set(report.results) == {"ed", "ed-k"}
    assert "U = 0" in report.skipped["free"]


@pytest.mark.parametrize("U", [0.0, 2.0, 8.0, 50.0])
@pytest.mark.parametrize("filling", [(2, 2), (3, 2), (2, 1)])
def test_ed_results_pass_sanity_checks(U, filling):
    m = HubbardModel(
        lattice=SquareLattice(lx=3, ly=2),
        t_prime=-0.25,
        U=U,
        filling=Filling(n_up=filling[0], n_dn=filling[1]),
    )
    checks = sanity_check(m, ExactDiagonalization().solve(m))
    assert len(checks) == 3
    assert all(c.passed for c in checks), checks


def test_sanity_checks_catch_unphysical_results():
    m = HubbardModel(lattice=SquareLattice(lx=2, ly=2), U=4.0, filling=Filling(n_up=2, n_dn=2))
    good = ExactDiagonalization().solve(m)
    too_low = replace(good, observables={**good.observables, "energy": Observable(-100.0)})
    bad_d = replace(good, observables={**good.observables, "double_occupancy": Observable(0.9)})
    assert not all(c.passed for c in sanity_check(m, too_low))
    assert not all(c.passed for c in sanity_check(m, bad_d))
    other = m.with_updates(U=5.0)
    assert not sanity_check(other, good)[0].passed  # fingerprint mismatch
