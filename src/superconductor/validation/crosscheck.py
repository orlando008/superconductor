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
    Both exact limits are also checked at finite temperature (energy, double occupancy, free
    energy, entropy), and the atomic limit at attractive U. Interacting 1D rings are checked
    against the exact Bethe ansatz at any U.
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
        # Finite temperature (ED's thermal path vs the canonical atomic limit), including
        # attractive U and over-half filling. Small lattices only: ED needs the full spectrum.
        if n <= 6:
            thermal_fillings = sorted(f for f in fillings if min(f) >= 0)
            thermal_fillings.append((n // 2 + 1, n // 2 + 1))
            for nu, nd in thermal_fillings:
                for temperature, U in ((0.3, 6.0), (2.0, 6.0), (0.5, -3.0)):
                    models.append(
                        HubbardModel(
                            name=f"atomic T={temperature} U={U:g} {lat.lx}x{lat.ly} ({nu},{nd})",
                            lattice=lat,
                            t=0.0,
                            U=U,
                            mu=0.2,
                            temperature=temperature,
                            filling=Filling(n_up=nu, n_dn=nd),
                        )
                    )
            # attractive U at T = 0: ground state maximizes double occupancy
            models.append(
                HubbardModel(
                    name=f"atomic T=0 U=-3 {lat.lx}x{lat.ly}",
                    lattice=lat,
                    t=0.0,
                    U=-3.0,
                    filling=Filling(n_up=n // 2, n_dn=max(n // 2 - 1, 0)),
                )
            )
            # Finite temperature at U = 0 (ED's thermal path vs canonical free fermions),
            # with diagonal hopping and a chemical potential.
            for nu, nd in thermal_fillings:
                for temperature in (0.3, 2.0):
                    models.append(
                        HubbardModel(
                            name=f"free T={temperature} {lat.lx}x{lat.ly} ({nu},{nd})",
                            lattice=lat,
                            t=1.0,
                            t_prime=-0.3,
                            U=0.0,
                            mu=0.2,
                            temperature=temperature,
                            filling=Filling(n_up=nu, n_dn=nd),
                        )
                    )
    # Interacting 1D rings: ED against the exact Bethe ansatz, at any U. Covers closed and
    # open shells, ground states with higher total spin, and (at L = 10) ED's Lanczos path.
    rings = [
        (4, [(1, 1), (2, 1), (2, 2)], (1.0, 6.0)),
        (6, [(2, 1), (2, 2), (3, 1), (3, 3)], (4.0,)),
        (6, [(3, 3)], (0.5, 12.0)),
        (8, [(2, 2), (3, 2)], (3.0,)),
        (10, [(5, 5)], (4.0,)),
    ]
    for length, fillings_1d, us in rings:
        for nu, nd in fillings_1d:
            for U in us:
                models.append(
                    HubbardModel(
                        name=f"ring L={length} ({nu},{nd}) U={U:g}",
                        lattice=SquareLattice(lx=length, ly=1),
                        t=1.0,
                        U=U,
                        mu=0.1,
                        filling=Filling(n_up=nu, n_dn=nd),
                    )
                )
    return models


def hamiltonian_suite() -> list[HubbardModel]:
    """Models whose ED Hamiltonian is checked against the independent reference construction.

    Generic interacting models (t, t', U, mu all non-zero) on every boundary type and on
    directions of length 1, 2 and 3, including the doubled-bond case of periodic length 2,
    plus the distinct physics of the standard suite (temperature doesn't change H).
    """
    from superconductor.validation.reference import MAX_REFERENCE_SITES

    lattices = [
        SquareLattice(lx=1, ly=3),
        SquareLattice(lx=2, ly=1),
        SquareLattice(lx=2, ly=1, boundary_x="open"),
        SquareLattice(lx=3, ly=1),
        SquareLattice(lx=2, ly=2),
        SquareLattice(lx=2, ly=2, boundary_x="open"),
        SquareLattice(lx=2, ly=3),
        SquareLattice(lx=3, ly=2, boundary_y="open"),
        SquareLattice(lx=2, ly=3, boundary_x="open", boundary_y="open"),
        SquareLattice(lx=4, ly=2),
    ]
    models: dict[str, HubbardModel] = {}
    for lat in lattices:
        n = lat.num_sites
        for nu, nd in {(1, 1), (n // 2, max(n // 2 - 1, 0)), (n // 2, n // 2), (n // 2 + 1, 1)}:
            if nu > n or nd > n:
                continue
            bc = lat.boundary_x[0] + lat.boundary_y[0]
            m = HubbardModel(
                name=f"generic {lat.lx}x{lat.ly} {bc} ({nu},{nd})",
                lattice=lat,
                t=1.0,
                t_prime=-0.35,
                U=5.5,
                mu=0.4,
                filling=Filling(n_up=nu, n_dn=nd),
            )
            models[m.fingerprint()] = m
    for m in standard_suite():
        if m.num_sites <= MAX_REFERENCE_SITES:
            ground = m.with_updates(temperature=0.0)
            models.setdefault(ground.fingerprint(), ground)
    return list(models.values())


def run_hamiltonian_checks(models: Iterable[HubbardModel] | None = None) -> list:
    """Compare ED's Hamiltonian with the independent reference for each model."""
    from superconductor.solvers.ed import build_hamiltonian
    from superconductor.validation.reference import compare_with

    return [compare_with(m, build_hamiltonian(m)[0]) for m in (models or hamiltonian_suite())]


def run_suite(models: Iterable[HubbardModel] | None = None, **kwargs) -> list[CrossCheckReport]:
    return [crosscheck(m, **kwargs) for m in (models or standard_suite())]
