"""Automatic cross-validation between solvers.

Wherever two methods should agree, run both and compare every shared observable. A
disagreement is either a bug or real physics (e.g. a method outside its regime); either way
it must be flagged, never silently averaged away.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from superconductor.model import Filling, HubbardModel, SquareLattice
from superconductor.solvers import Result, Solver, get_solver, list_solvers


@dataclass(frozen=True)
class Comparison:
    observable: str
    solver_a: str
    solver_b: str
    value_a: float
    value_b: float
    tolerance: float

    @property
    def difference(self) -> float:
        return abs(self.value_a - self.value_b)

    @property
    def passed(self) -> bool:
        return self.difference <= self.tolerance


@dataclass
class CrossCheckReport:
    model: HubbardModel
    results: dict[str, Result] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.comparisons)

    @property
    def failures(self) -> list[Comparison]:
        return [c for c in self.comparisons if not c.passed]


def compare_results(a: Result, b: Result, atol: float = 1e-8, rtol: float = 1e-8):
    """Compare all observables two results share. Error bars widen the tolerance (2 sigma)."""
    if a.model_fingerprint != b.model_fingerprint:
        raise ValueError("Results come from different models and cannot be compared")
    out = []
    for name in sorted(a.observables.keys() & b.observables.keys()):
        oa, ob = a.observables[name], b.observables[name]
        scale = max(abs(oa.value), abs(ob.value))
        tol = atol + rtol * scale + 2.0 * (oa.error**2 + ob.error**2) ** 0.5
        out.append(Comparison(name, a.solver, b.solver, oa.value, ob.value, tol))
    return out


def crosscheck(
    model: HubbardModel,
    solvers: Sequence[Solver] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> CrossCheckReport:
    """Run every applicable solver on ``model`` and compare all pairs."""
    solvers = solvers if solvers is not None else [get_solver(n) for n in list_solvers()]
    report = CrossCheckReport(model=model)
    for solver in solvers:
        reason = solver.applicability(model)
        if reason is not None:
            report.skipped[solver.name] = reason
            continue
        report.results[solver.name] = solver.solve(model)
    for a, b in combinations(report.results.values(), 2):
        report.comparisons.extend(compare_results(a, b, atol=atol, rtol=rtol))
    return report


def standard_suite() -> list[HubbardModel]:
    """Models where independent methods must agree exactly.

    Covers U = 0 (ED vs free fermions) and t = t' = 0 (ED vs atomic limit) across lattice
    shapes, boundary conditions, fillings, open and closed shells, and nonzero t' and mu.
    """
    lattices = [
        SquareLattice(lx=2, ly=1, boundary_x="open"),
        SquareLattice(lx=4, ly=1),
        SquareLattice(lx=2, ly=2),
        SquareLattice(lx=3, ly=2, boundary_y="open"),
        SquareLattice(lx=4, ly=2),
    ]
    models: list[HubbardModel] = []
    for lat in lattices:
        n = lat.num_sites
        fillings = {(n // 2, n // 2), (n // 2, n // 2 - 1), (max(n // 2 - 1, 0), n // 2 - 1)}
        if n % 2:
            fillings.add(((n + 1) // 2, n // 2))
        for nu, nd in sorted(f for f in fillings if min(f) >= 0):
            filling = Filling(n_up=nu, n_dn=nd)
            label = f"{lat.lx}x{lat.ly} {lat.boundary_x[0]}{lat.boundary_y[0]} ({nu},{nd})"
            models.append(
                HubbardModel(
                    name=f"free {label}",
                    lattice=lat,
                    t=1.0,
                    t_prime=-0.3,
                    mu=0.2,
                    U=0.0,
                    filling=filling,
                )
            )
            models.append(
                HubbardModel(name=f"atomic {label}", lattice=lat, t=0.0, U=6.0, filling=filling)
            )
        # an over-half-filled atomic case, where double occupancy is forced
        models.append(
            HubbardModel(
                name=f"atomic forced-double {lat.lx}x{lat.ly}",
                lattice=lat,
                t=0.0,
                U=4.0,
                filling=Filling(n_up=n // 2 + 1, n_dn=n // 2 + 1),
            )
        )
    return models


def run_suite(models: Iterable[HubbardModel] | None = None, **kwargs) -> list[CrossCheckReport]:
    return [crosscheck(m, **kwargs) for m in (models or standard_suite())]
