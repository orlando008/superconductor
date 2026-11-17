"""Cost estimates (Solver.estimate) and the CLI around them.

Wall time depends on the hardware, so time accuracy is checked by benchmarks/ed_cost.py,
run deliberately. These tests check what doesn't depend on the machine: matrix sparsity,
which method ``solve`` will use, memory, feasibility, and the CLI's budget warnings.
"""

import json
import time
import tracemalloc

import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.cli import budget_warnings, main
from superconductor.solvers import (
    AtomicLimit,
    CostEstimate,
    ExactDiagonalization,
    FreeFermions,
    Observable,
    Solver,
    standard_observables,
)
from superconductor.solvers.ed import build_hamiltonian, estimated_nonzeros

ED = ExactDiagonalization()


def model(lx, ly, nu, nd, *, t=1.0, tp=-0.3, U=6.0, T=0.0):
    return HubbardModel(
        lattice=SquareLattice(lx=lx, ly=ly),
        t=t,
        t_prime=tp if t else 0.0,
        U=U,
        temperature=T,
        filling=Filling(n_up=nu, n_dn=nd),
    )


@pytest.mark.parametrize(
    "m",
    [
        model(3, 2, 3, 3),
        model(4, 2, 3, 3),
        model(4, 2, 4, 4),
        model(3, 3, 4, 4),
        model(4, 1, 2, 1),  # small: some diagonal entries are exactly zero
        model(4, 2, 4, 4, t=0.0),  # no hopping at all
        model(3, 2, 2, 2, U=0.0),  # U = 0 and mu = 0: the whole diagonal is zero
    ],
)
def test_nonzero_count_matches_built_hamiltonian(m):
    """The counting formula gives exactly the number of non-zeros in the built matrix."""
    ham, _ = build_hamiltonian(m)
    assert estimated_nonzeros(m) == ham.nnz


@pytest.mark.parametrize(
    "m,estimate_method,solve_method",
    [
        (model(3, 2, 3, 3), "dense", "dense"),
        (model(4, 2, 4, 4), "Lanczos", "lanczos+deflation"),
        (model(4, 2, 4, 4, t=0.0), "diagonal", "diagonal"),
        (model(3, 2, 3, 3, T=1.0), "dense (full spectrum", "dense-thermal"),
    ],
)
def test_estimate_predicts_the_path_solve_takes(m, estimate_method, solve_method):
    est = ED.estimate(m)
    assert est.details["method"].startswith(estimate_method)
    assert ED.solve(m).metadata["method"] == solve_method


@pytest.mark.parametrize(
    "m",
    [
        model(4, 2, 3, 2),  # dense, about 100 MB
        model(3, 3, 4, 4),  # Lanczos
        model(4, 2, 4, 4, t=0.0),  # diagonal with a 70-fold degenerate ground state
        model(3, 2, 3, 3, T=1.0),  # finite temperature
    ],
)
def test_memory_estimate_within_factor_three(m):
    est = ED.estimate(m)
    tracemalloc.start()
    ED.solve(m)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak / 3 <= est.memory_bytes <= peak * 3, (est.memory_bytes, peak)


def test_out_of_reach_model_is_reported_quickly_with_numbers():
    """4x4 at half filling: 1.7e8 states. The estimate says no, and how far out of reach."""
    m = model(4, 4, 8, 8, U=8.0)
    start = time.perf_counter()
    est = ED.estimate(m)
    assert time.perf_counter() - start < 1.0
    assert est.feasible is False and "exceeds limit" in est.reason
    assert est.details["sector_dimension"] == 165_636_900
    assert est.memory_bytes > 1e11 and est.seconds > 3600


def test_estimates_grow_with_problem_size_within_each_method():
    """Within one method, bigger costs more. Across methods it needn't: Lanczos on 15,876
    states really is faster than dense diagonalization of 3,136 (about 1 s against 4 s)."""
    dense = [model(2, 2, 2, 2), model(3, 2, 3, 2), model(4, 2, 3, 2), model(4, 2, 3, 3)]
    lanczos = [model(4, 2, 4, 4), model(3, 3, 4, 4), model(4, 3, 4, 4), model(4, 3, 5, 5)]
    for group, method in ((dense, "dense"), (lanczos, "Lanczos")):
        estimates = [ED.estimate(m) for m in group]
        assert all(e.details["method"].startswith(method) for e in estimates)
        for key in ("seconds", "memory_bytes"):
            values = [getattr(e, key) for e in estimates]
            assert values == sorted(values), (method, key)
    assert ED.estimate(lanczos[1]).seconds < ED.estimate(dense[-1]).seconds


def test_default_estimate_knows_nothing_but_applicability():
    class Toy(Solver):
        name = "toy"

        def applicability(self, m):
            return None if m.U == 0 else "needs U = 0"

        def _solve(self, m):
            return standard_observables(m, 0.0, 0.0), {}

    ok = Toy().estimate(model(2, 2, 1, 1, U=0.0))
    assert ok.feasible is None and ok.seconds is None and ok.memory_bytes is None
    no = Toy().estimate(model(2, 2, 1, 1))
    assert no.feasible is False and no.reason == "needs U = 0"
    assert isinstance(Observable(1.0), Observable)


@pytest.mark.parametrize("solver,applicable", [(FreeFermions(), "U"), (AtomicLimit(), "t")])
def test_exact_limit_estimates(solver, applicable):
    good = model(10, 10, 45, 45, U=0.0) if applicable == "U" else model(10, 10, 45, 45, t=0.0)
    est = solver.estimate(good)
    assert est.feasible is True and est.seconds < 1.0 and est.memory_bytes < 1e8
    bad = solver.estimate(model(2, 2, 1, 1))
    assert bad.feasible is False and bad.reason


def test_budget_warnings():
    est = CostEstimate(solver="x", feasible=True, seconds=100.0, memory_bytes=2e9, driver="d")
    assert budget_warnings(est, seconds=1000, gigabytes=8) == []
    assert len(budget_warnings(est, seconds=10, gigabytes=1)) == 2
    unknown = CostEstimate(solver="x", feasible=None)
    assert budget_warnings(unknown, seconds=0, gigabytes=0) == []


@pytest.fixture
def model_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(model(3, 2, 3, 3).to_json())
    return str(p)


def test_cli_estimate_text_and_json(model_file, capsys):
    assert main(["estimate", model_file]) == 0
    out = capsys.readouterr().out
    assert "Can run:    yes" in out and "sector dimension 400" in out
    assert main(["estimate", model_file, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["feasible"] is True and data["details"]["sector_dimension"] == 400
    assert data["over_budget"] == []


def test_solve_warns_but_still_runs_when_over_budget(model_file, capsys):
    assert main(["solve", model_file, "--budget-seconds", "0", "--budget-gb", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.err.count("Warning:") == 2
    assert json.loads(captured.out)["solver"] == "ed"


def test_budget_from_environment(model_file, capsys, monkeypatch):
    monkeypatch.setenv("SUPERCONDUCTOR_BUDGET_SECONDS", "0")
    assert main(["estimate", model_file]) == 0
    assert "exceeds the budget" in capsys.readouterr().out
