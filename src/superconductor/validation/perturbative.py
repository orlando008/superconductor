"""Perturbative limits as cross-checks: ED must approach the exact limits at the right rate.

The exact-limit solvers check ED *at* U = 0 and t = 0. These checks test it *near* those
limits, which catches errors that vanish exactly at the limit (an interaction term with the
wrong prefactor, for example, is invisible at U = 0).

**Weak coupling.** If the U = 0 ground state is non-degenerate (a closed shell), first-order
perturbation theory (equivalently Hellmann-Feynman at U = 0) gives

    E0(U) = E_free + U * D_free + r(U),   r(U) = O(U^2),

with D_free the total double occupancy of the free ground state. The variational principle
also makes the remainder rigorously non-positive: r(U) <= 0 for every U >= 0. Both are
checked: r <= 0, and log|r| against log U has slope 2.

**Strong coupling.** At half filling (one electron per site), second-order perturbation
theory in t/U maps the low-energy physics onto the Heisenberg model,

    H_eff = sum_{i<j} J_ij (S_i . S_j - 1/4),   J_ij = 4 h_ij^2 / U,

with h_ij the total hopping between sites i and j (so diagonal hopping gives J' = 4 t'^2/U),
and E0(U) = E_Heisenberg(U) - mu * N + O(t^4 / U^3). The slope of log|remainder| against
log U is checked to be -3.

Reference quantities come from the independent construction in ``reference.py``, and the
Heisenberg ground state from a small exact spin solver in this module, so the only code
under test is the ED solver itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from superconductor.model import HubbardModel
from superconductor.validation.reference import reference_free_ground_state, reference_hopping

WEAK_U = (0.005, 0.01, 0.02, 0.04)
STRONG_U = (25.0, 50.0, 100.0, 200.0)
WEAK_EXPONENT, STRONG_EXPONENT = 2.0, -3.0
EXPONENT_TOLERANCE = 0.15
_SHELL_TOL = 1e-9


@dataclass(frozen=True)
class PerturbativeOutcome:
    name: str
    model_name: str | None
    status: Literal["pass", "fail", "skipped"]
    detail: str
    fitted_exponent: float | None = None
    remainders: tuple[float, ...] = field(default=())


# ----- an exact Heisenberg solver (reference only) --------------------------------------------


def heisenberg_ground_energy(
    couplings: dict[tuple[int, int], float], n_sites: int, n_up: int
) -> float:
    """Lowest eigenvalue of sum_{i<j} J_ij S_i . S_j in the sector with ``n_up`` up spins.

    Spins only, so no fermion signs: basis states are bit strings (1 = up), the diagonal is
    J S^z S^z and the off-diagonal term (J/2)(S+S- + S-S+) swaps antiparallel neighbours.
    """
    states = [s for s in range(1 << n_sites) if bin(s).count("1") == n_up]
    index = {s: k for k, s in enumerate(states)}
    dim = len(states)
    rows, cols, vals = [], [], []
    diag = np.zeros(dim)
    for k, s in enumerate(states):
        for (i, j), coupling in couplings.items():
            si, sj = (s >> i) & 1, (s >> j) & 1
            diag[k] += coupling * (0.25 if si == sj else -0.25)
            if si != sj:
                rows.append(index[s ^ (1 << i) ^ (1 << j)])
                cols.append(k)
                vals.append(0.5 * coupling)
    ham = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim)) + sp.diags(diag)
    if dim <= 4096:
        return float(np.linalg.eigvalsh(ham.toarray())[0])
    return float(eigsh(ham, k=1, which="SA", tol=1e-13)[0][0])


def heisenberg_couplings(model: HubbardModel, U: float) -> dict[tuple[int, int], float]:
    """J_ij = 4 h_ij^2 / U for every pair of sites connected by hopping (independent h)."""
    h = reference_hopping(model)
    n = model.num_sites
    return {(i, j): 4.0 * h[i, j] ** 2 / U for i in range(n) for j in range(i + 1, n) if h[i, j]}


# ----- the checks -------------------------------------------------------------------------------

Energy = Callable[[HubbardModel], float]


def _ed_energy(model: HubbardModel) -> float:
    from superconductor.solvers import ExactDiagonalization

    return ExactDiagonalization().solve(model)["energy"]


def _fit_exponent(us: Sequence[float], remainders: Sequence[float]) -> float:
    return float(np.polyfit(np.log(us), np.log(np.abs(remainders)), 1)[0])


def _closed_shell(model: HubbardModel) -> bool:
    levels = np.linalg.eigvalsh(reference_hopping(model.with_updates(U=0.0)))
    for n in (model.filling.n_up, model.filling.n_dn):
        if 0 < n < len(levels) and levels[n] - levels[n - 1] <= _SHELL_TOL * max(
            1.0, abs(levels[n])
        ):
            return False
    return True


def weak_coupling_check(
    model: HubbardModel, energy: Energy = _ed_energy, us: Sequence[float] = WEAK_U
) -> PerturbativeOutcome:
    name = "weak_coupling"
    f = model.filling
    if model.temperature != 0:
        return PerturbativeOutcome(name, model.name, "skipped", "ground-state check; T > 0")
    if f.n_up == 0 or f.n_dn == 0:
        return PerturbativeOutcome(
            name, model.name, "skipped", "one spin species is empty, so U has no effect"
        )
    if not _closed_shell(model):
        return PerturbativeOutcome(
            name,
            model.name,
            "skipped",
            "free ground state is degenerate (open shell); degenerate perturbation theory "
            "is out of scope",
        )
    e_free, d_free = reference_free_ground_state(model.with_updates(U=0.0))
    remainders = tuple(energy(model.with_updates(U=u)) - (e_free + u * d_free) for u in us)
    scale = max(1.0, abs(e_free))
    if any(r > 1e-10 * scale for r in remainders):
        return PerturbativeOutcome(
            name,
            model.name,
            "fail",
            f"energy above the variational bound E_free + U D_free: remainders {remainders}",
            None,
            remainders,
        )
    exponent = _fit_exponent(us, remainders)
    ok = abs(exponent - WEAK_EXPONENT) <= EXPONENT_TOLERANCE
    detail = f"remainder ~ U^{exponent:.3f} (expected U^2)"
    return PerturbativeOutcome(
        name, model.name, "pass" if ok else "fail", detail, exponent, remainders
    )


def strong_coupling_check(
    model: HubbardModel, energy: Energy = _ed_energy, us: Sequence[float] = STRONG_U
) -> PerturbativeOutcome:
    name = "strong_coupling"
    f = model.filling
    if model.temperature != 0:
        return PerturbativeOutcome(name, model.name, "skipped", "ground-state check; T > 0")
    if f.total != model.num_sites:
        return PerturbativeOutcome(
            name,
            model.name,
            "skipped",
            "Heisenberg limit needs half filling (one electron per site)",
        )
    if not heisenberg_couplings(model, 1.0):
        return PerturbativeOutcome(name, model.name, "skipped", "no hopping, so no spin exchange")
    remainders = []
    for u in us:
        e_heis = heisenberg_ground_energy(heisenberg_couplings(model, u), model.num_sites, f.n_up)
        constant = -sum(heisenberg_couplings(model, u).values()) / 4.0 - model.mu * f.total
        remainders.append(energy(model.with_updates(U=u)) - (e_heis + constant))
    remainders = tuple(remainders)
    exponent = _fit_exponent(us, remainders)
    ok = abs(exponent - STRONG_EXPONENT) <= EXPONENT_TOLERANCE
    detail = f"remainder ~ U^{exponent:.3f} (expected U^-3)"
    return PerturbativeOutcome(
        name, model.name, "pass" if ok else "fail", detail, exponent, remainders
    )


def perturbative_suite() -> list[HubbardModel]:
    """Models for both checks: closed shells for weak coupling, half filling for strong."""
    from superconductor.model import Filling, SquareLattice

    cases = [
        (SquareLattice(lx=2, ly=1, boundary_x="open"), (1, 1)),  # the two-site dimer
        (SquareLattice(lx=4, ly=1), (2, 2)),
        (SquareLattice(lx=4, ly=1), (1, 1)),
        (SquareLattice(lx=2, ly=2), (2, 2)),
        (SquareLattice(lx=2, ly=2), (1, 1)),
        (SquareLattice(lx=3, ly=2), (3, 3)),
        (SquareLattice(lx=3, ly=2, boundary_y="open"), (2, 2)),
        (SquareLattice(lx=3, ly=2, boundary_y="open"), (3, 3)),
        (SquareLattice(lx=4, ly=2), (1, 1)),
        (SquareLattice(lx=4, ly=2), (4, 4)),
    ]
    out = []
    for lat, (nu, nd) in cases:
        bc = lat.boundary_x[0] + lat.boundary_y[0]
        for tp in (0.0, -0.3):
            out.append(
                HubbardModel(
                    name=f"{lat.lx}x{lat.ly} {bc} ({nu},{nd}) t'={tp}",
                    lattice=lat,
                    t=1.0,
                    t_prime=tp,
                    U=1.0,
                    mu=0.25,
                    filling=Filling(n_up=nu, n_dn=nd),
                )
            )
    return out


def run_perturbative_checks(
    models: Sequence[HubbardModel] | None = None, energy: Energy = _ed_energy
) -> list[PerturbativeOutcome]:
    out = []
    for m in models or perturbative_suite():
        out.append(weak_coupling_check(m, energy))
        out.append(strong_coupling_check(m, energy))
    return out
