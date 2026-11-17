"""The independent reference Hamiltonian, and proof that it closes the shared-code blind spot."""

import ast
import inspect

import numpy as np
import pytest

import superconductor.validation.reference as reference_module
import superconductor.validation.sanity as sanity_module
from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.model.hubbard import HubbardModel as _HubbardModel
from superconductor.solvers import FreeFermions
from superconductor.solvers.ed import build_hamiltonian
from superconductor.validation import (
    compare_with,
    crosscheck,
    hamiltonian_suite,
    reference_free_ground_state,
    reference_hopping,
    run_hamiltonian_checks,
)
from superconductor.validation.reference import reference_hamiltonian


def model(lattice, nu, nd, **kw):
    params = dict(t=1.0, t_prime=-0.35, U=5.5, mu=0.4) | kw
    return HubbardModel(lattice=lattice, filling=Filling(n_up=nu, n_dn=nd), **params)


# Lengths 1, 2 and 3 in each direction, periodic and open.
LATTICES = [
    SquareLattice(lx=1, ly=3),
    SquareLattice(lx=3, ly=1, boundary_x="open"),
    SquareLattice(lx=2, ly=1),
    SquareLattice(lx=2, ly=2),
    SquareLattice(lx=2, ly=2, boundary_y="open"),
    SquareLattice(lx=3, ly=2),
    SquareLattice(lx=2, ly=3, boundary_x="open", boundary_y="open"),
    SquareLattice(lx=3, ly=3),
    SquareLattice(lx=4, ly=2, boundary_x="open"),
]


# ---- independence guards ---------------------------------------------------------------------


def _calls_and_imports(module):
    tree = ast.parse(inspect.getsource(module))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return calls, imports


def test_reference_shares_no_code_with_solvers_or_lattice_helpers():
    calls, imports = _calls_and_imports(reference_module)
    assert "hopping_matrix" not in calls and "bonds" not in calls
    assert not any(m.startswith("superconductor.solvers") for m in imports)
    assert not any(m.startswith("superconductor.model.lattice") for m in imports)


def test_sanity_checks_do_not_run_solver_code():
    calls, imports = _calls_and_imports(sanity_module)
    assert "solve" not in calls
    # The only solvers import is the type-only one under TYPE_CHECKING.
    source = inspect.getsource(sanity_module)
    assert "from superconductor.solvers import FreeFermions" not in source


# ---- agreement with the solver path -------------------------------------------------------------


@pytest.mark.parametrize("lattice", LATTICES)
def test_hopping_by_image_counting_matches_the_lattice_code(lattice):
    """Two independent algorithms for the same documented convention."""
    m = model(lattice, 1, 1)
    assert np.allclose(reference_hopping(m), m.hopping_matrix(), atol=0)


@pytest.mark.parametrize("lattice", LATTICES[:7])
@pytest.mark.parametrize("filling", [(1, 1), (2, 1), (1, 0), (2, 2)])
def test_ed_hamiltonian_matches_reference_spectrum(lattice, filling):
    nu, nd = filling
    if max(nu, nd) > lattice.num_sites:
        pytest.skip("filling exceeds lattice")
    m = model(lattice, nu, nd)
    check = compare_with(m, build_hamiltonian(m)[0])
    assert check.passed, check.detail
    assert check.max_spectrum_difference is not None and check.max_spectrum_difference < 1e-9


def test_reference_double_occupancy_operator():
    """Tr D over a sector equals sum_d d * g(d): an independent count (no hopping involved)."""
    from math import comb

    m = model(SquareLattice(lx=2, ly=2), 2, 2)
    ref = reference_hamiltonian(m)
    n, nu, nd = 4, 2, 2
    expected = sum(
        d * comb(n, d) * comb(n - d, nu - d) * comb(n - nu, nd - d)
        for d in range(0, min(nu, nd) + 1)
    )
    assert ref.double_occupancy.diagonal().sum() == pytest.approx(expected)


def test_full_hamiltonian_suite_passes():
    checks = run_hamiltonian_checks()
    assert len(checks) == len(hamiltonian_suite()) >= 100
    assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]


def test_free_reference_matches_free_solver_including_open_shells():
    for lattice, nu, nd in [
        (SquareLattice(lx=2, ly=2), 2, 2),  # open shell
        (SquareLattice(lx=4, ly=1), 2, 1),
        (SquareLattice(lx=10, ly=10), 43, 41),
        (SquareLattice(lx=3, ly=2, boundary_y="open"), 3, 2),
    ]:
        m = model(lattice, nu, nd, U=0.0)
        e, d = reference_free_ground_state(m)
        free = FreeFermions().solve(m)
        assert e == pytest.approx(free["energy"], abs=1e-10)
        assert d / m.num_sites == pytest.approx(free["double_occupancy"], abs=1e-12)


# ---- mutation tests: the reason this check exists ----------------------------------------------

_original_hopping = _HubbardModel.hopping_matrix


def _flipped_t_prime(self):
    """A plausible bug: the diagonal hopping enters with the wrong sign."""
    h = _original_hopping(self)
    if self.t_prime == 0:
        return h
    return _original_hopping(self.with_updates(t_prime=-self.t_prime))


def _dropped_wraparound(self):
    """A plausible bug: periodic wrap-around bonds are missing."""
    h = _original_hopping(self).copy()
    lx = self.lattice.lx
    for i in range(self.num_sites):
        for j in range(self.num_sites):
            if abs(i % lx - j % lx) == lx - 1 and lx > 2:
                h[i, j] = 0.0
    return h


def _solver_disagreements(report):
    return {frozenset((c.solver_a, c.solver_b)) for c in report.failures}


def test_translation_invariant_hopping_bug_fools_every_solver(monkeypatch):
    """A flipped t' sign keeps the lattice translation-invariant, so ED, momentum ED and the
    free solver all inherit it and agree with each other. Only the reference catches it."""
    lattice = SquareLattice(lx=3, ly=2)
    free_model = model(lattice, 2, 2, U=0.0)
    interacting = model(lattice, 2, 2)
    monkeypatch.setattr(_HubbardModel, "hopping_matrix", _flipped_t_prime)

    report = crosscheck(free_model)
    assert set(report.results) == {"ed", "ed-k", "free"}
    assert report.passed  # the blind spot

    for m in (free_model, interacting):
        assert not compare_with(m, build_hamiltonian(m)[0]).passed


def test_translation_breaking_hopping_bug_is_caught_twice(monkeypatch):
    """Dropped wrap-around bonds break translation symmetry. Momentum ED assumes that
    symmetry, so it now disagrees with ED and the free solver (ED and free still agree with
    each other, sharing the bug). The reference catches it as well."""
    lattice = SquareLattice(lx=3, ly=2)
    free_model = model(lattice, 2, 2, U=0.0)
    interacting = model(lattice, 2, 2)
    monkeypatch.setattr(_HubbardModel, "hopping_matrix", _dropped_wraparound)

    report = crosscheck(free_model)
    assert not report.passed
    disagreements = _solver_disagreements(report)
    assert frozenset(("ed", "free")) not in disagreements
    assert disagreements <= {frozenset(("ed", "ed-k")), frozenset(("ed-k", "free"))}

    for m in (free_model, interacting):
        assert not compare_with(m, build_hamiltonian(m)[0]).passed
