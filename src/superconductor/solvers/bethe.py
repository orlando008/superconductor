"""Exact ground states of periodic 1D Hubbard chains from the Lieb-Wu Bethe ansatz.

The 1D Hubbard model is integrable (Lieb and Wu, Phys. Rev. Lett. 20, 1445 (1968); see
Essler et al., *The One-Dimensional Hubbard Model*, Cambridge 2005, ch. 3 and 4). For a
periodic ring of L sites with N electrons, M of them spin down, every eigenstate that is a
highest-weight state of SU(2) spin (total spin S = N/2 - M, S^z = S) is described by N
charge momenta k_j and M spin rapidities lambda_a solving, in logarithmic form with
u = U / (4t) and theta(x) = 2 arctan(x),

    L k_j = 2 pi I_j - sum_a theta((sin k_j - lambda_a) / u),
    sum_j theta((lambda_a - sin k_j) / u) = 2 pi J_a + sum_b theta((lambda_a - lambda_b) / (2u)),

with energy E = -2t sum_j cos k_j (including the interaction). The quantum numbers are
I_j in Z + M/2 and J_a in Z + (N - M + 1)/2.

Finding the *ground state* of a sector takes three facts, each checked against exact
diagonalization in the tests:

1. **Spin multiplets.** A sector (N_up, N_dn) holds every spin multiplet with
   S >= |S^z|, not just S = |S^z|. Its ground state is the lowest highest-weight state over
   M' = 0, 1, ..., min(N_up, N_dn). At strong coupling it can have higher spin than the sector
   suggests. Example: the 6-site ring with 3 electrons at U = 12 has a fully polarized
   ground state with energy exactly -4, with no double occupancy at all.
2. **Shell effects.** When N consecutive I_j (or M consecutive J_a) can't be centred on zero
   with the required parity, which happens for "open shells" like N = 4n on a ring, the
   ground state is not unique among nearby patterns. We search all N-element subsets of the
   N + 2 allowed values closest to zero (likewise for J), solve each, and keep the lowest.
3. **Small U.** As U -> 0 the equations become step-like, and a direct solve can land on the
   wrong root. Below u = 0.5 we continue from u = 0.5 down to the target in small steps.

Every converged solution with distinct k_j and distinct lambda_a is a genuine eigenstate, so
the lowest one found is an **upper bound** on the true ground energy, equal to it whenever
the ground state's pattern is among those searched. The search was verified to be exact
against ED for every filling on rings of 2 to 8 sites, and for selected 10-site rings, at
U = 0.3, 4 and 15.

Double occupancy follows from Hellmann-Feynman, D = dE/dU, by a Richardson-extrapolated
central difference that reuses the ground state's quantum numbers. Its error estimate goes
into ``Observable.error``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import root

from superconductor.model import HubbardModel
from superconductor.solvers.base import CostEstimate, Observable, Solver, standard_observables

MAX_SITES = 24
WINDOW_EXTRA = 2  # search subsets of the n + 2 allowed quantum numbers closest to zero
CONTINUATION_START = 0.5  # u below which the solve is continued down from here
CONTINUATION_STEPS = 12
RESIDUAL_TOL = 1e-11
DISTINCT_TOL = 1e-7
FD_STEP = 1e-3  # relative step for the Hellmann-Feynman derivative


def _theta(x):
    return 2.0 * np.arctan(x)


def _dtheta(x):
    return 2.0 / (1.0 + x * x)


def _equations(x, L, N, M, u, I, J):
    k, lam = x[:N], x[N:]
    s, c = np.sin(k), np.cos(k)
    a = (s[:, None] - lam[None, :]) / u  # N x M
    b = (lam[:, None] - lam[None, :]) / (2 * u)  # M x M
    f = L * k - 2 * np.pi * I + _theta(a).sum(1)
    g = _theta(-a).sum(0) - 2 * np.pi * J - _theta(b).sum(1)
    jac = np.zeros((N + M, N + M))
    da, db = _dtheta(a), _dtheta(b)
    jac[:N, :N] = np.diag(L + (da.sum(1) * c) / u)
    jac[:N, N:] = -da / u
    jac[N:, :N] = -(da * c[:, None]).T / u
    jac[N:, N:] = np.diag(da.sum(0) / u - (db.sum(1) - 1.0) / (2 * u)) + (
        db - np.diag(np.diag(db))
    ) / (2 * u)
    return np.concatenate([f, g]), jac


def _root(x0, args):
    def fun(x):
        return _equations(x, *args)

    for method in ("hybr", "lm"):
        sol = root(fun, x0, jac=True, method=method, tol=1e-14)
        if np.max(np.abs(fun(sol.x)[0]), initial=0.0) < RESIDUAL_TOL:
            return sol.x
    return None


def solve_bethe(L: int, N: int, M: int, u: float, I, J) -> np.ndarray | None:
    """Roots (k, lambda) for one set of quantum numbers, or None if no valid solution."""
    I, J = np.asarray(I, float), np.asarray(J, float)
    us = (
        [u]
        if u >= CONTINUATION_START
        else list(np.geomspace(CONTINUATION_START, u, CONTINUATION_STEPS))
    )
    lam0 = us[0] * np.tan(np.clip(np.pi * J / max(N - M + 1, 1), -1.4, 1.4)) if M else np.zeros(0)
    x = np.concatenate([2 * np.pi * I / L, lam0])
    for step in us:
        x = _root(x, (L, N, M, step, I, J))
        if x is None:
            return None
    k, lam = x[:N], x[N:]
    if N > 1:
        dk = np.abs(np.angle(np.exp(1j * (k[:, None] - k[None, :])))) + 9 * np.eye(N)
        if dk.min() < DISTINCT_TOL:
            return None
    if M > 1 and (np.abs(lam[:, None] - lam[None, :]) + 9 * np.eye(M)).min() < DISTINCT_TOL:
        return None
    return x


def quantum_number_window(n: int, offset: float, extra: int = WINDOW_EXTRA) -> list[np.ndarray]:
    """All n-element subsets of the n + extra values in Z + offset closest to zero."""
    if n == 0:
        return [np.zeros(0)]
    c = np.arange(n + extra + 2) - (n + extra + 1) / 2
    c = c + ((offset - c[0]) % 1)
    values = sorted(sorted(c, key=lambda v: (abs(v), v))[: n + extra])
    return [np.array(s) for s in combinations(values, n)]


def _centred(n: int, offset: float) -> np.ndarray | None:
    """n consecutive values centred on zero, if that's compatible with Z + offset."""
    values = np.arange(n) - (n - 1) / 2
    return values if n == 0 or abs((values[0] - offset) % 1) < 1e-9 else None


def half_filled_closed_shell(L: int, N: int, M: int) -> bool:
    """Half filling with S^z = 0 on a ring of L = 4n + 2 sites.

    There the ground state is known without searching: it's a spin singlet (Lieb-Mattis, for
    an even ring), so M' = M, and the centred quantum numbers are allowed by the parity rules.
    That centred state is Lieb and Wu's half-filled ground state.
    """
    return N == L and L % 4 == 2 and 2 * M == N


def ground_state(L: int, N: int, M: int, U: float, t: float = 1.0, search: bool | None = None):
    """Lowest Bethe state over spin multiplets M' <= M and the quantum-number search.

    Returns (energy, M', I, J) without the chemical-potential term. ``search=False`` forces
    the closed-shell shortcut (only valid where ``half_filled_closed_shell`` holds);
    ``search=True`` forces the full search; the default uses the shortcut when valid.
    """
    u = U / (4 * t)
    if search is None:
        search = not half_filled_closed_shell(L, N, M)
    if not search:
        I, J = _centred(N, (M / 2) % 1), _centred(M, ((N - M + 1) / 2) % 1)
        x = solve_bethe(L, N, M, u, I, J)
        return None if x is None else (float(-2 * t * np.cos(x[:N]).sum()), M, I, J)
    best = None
    for mp in range(M + 1):
        for I in quantum_number_window(N, (mp / 2) % 1):
            for J in quantum_number_window(mp, ((N - mp + 1) / 2) % 1):
                x = solve_bethe(L, N, mp, u, I, J)
                if x is None:
                    continue
                e = float(-2 * t * np.cos(x[:N]).sum())
                if best is None or e < best[0] - 1e-12:
                    best = (e, mp, I, J)
    return best


def _energy_at(L, N, mp, U, t, I, J) -> float:
    x = solve_bethe(L, N, mp, U / (4 * t), I, J)
    if x is None:
        raise RuntimeError("Bethe state lost while differentiating in U")
    return float(-2 * t * np.cos(x[:N]).sum())


class BetheAnsatz(Solver):
    """Exact ground state of a periodic 1D Hubbard chain (t' = 0, U > 0, at most half filled)."""

    name = "bethe"
    version = "0.1.0"

    def applicability(self, model: HubbardModel) -> str | None:
        lat = model.lattice
        if lat.ly != 1:
            return "requires a 1D chain (ly = 1)"
        if lat.boundary_x != "periodic" or lat.lx < 2:
            return "requires a periodic ring of at least 2 sites"
        if lat.lx > MAX_SITES:
            return f"chains longer than {MAX_SITES} sites not supported yet"
        if model.t_prime != 0.0:
            return "requires t' = 0 (next-nearest hopping breaks integrability)"
        if model.t <= 0.0:
            return "requires t > 0"
        if model.U <= 0.0:
            return "requires U > 0 (use the free solver at U = 0)"
        if model.temperature != 0.0:
            return "ground state only"
        if model.filling.total > lat.lx:
            return "at most half filling (N <= L)"
        return None

    def settings(self) -> dict[str, Any]:
        return {
            "window_extra": WINDOW_EXTRA,
            "continuation_start_u": CONTINUATION_START,
            "continuation_steps": CONTINUATION_STEPS,
            "residual_tol": RESIDUAL_TOL,
            "distinct_tol": DISTINCT_TOL,
            "fd_step": FD_STEP,
        }

    def estimate(self, model: HubbardModel) -> CostEstimate:
        reason = self.applicability(model)
        n, m = model.filling.total, min(model.filling.n_up, model.filling.n_dn)
        e = WINDOW_EXTRA
        patterns = (
            1
            if half_filled_closed_shell(model.lattice.lx, n, m)
            else sum(
                len(list(combinations(range(n + e), n)))
                * len(list(combinations(range(mp + e), mp)))
                for mp in range(m + 1)
            )
        )
        return CostEstimate(
            solver=self.name,
            feasible=reason is None,
            seconds=0.02 + 2e-3 * patterns * (1 + (n + m) / 10),
            memory_bytes=1e6,
            driver=f"{patterns:,} quantum-number patterns to search",
            reason=reason,
            details={"patterns": patterns},
        )

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        L, t, U = model.lattice.lx, model.t, model.U
        N = model.filling.total
        M = min(model.filling.n_up, model.filling.n_dn)
        found = ground_state(L, N, M, U, t)
        if found is None:
            raise RuntimeError("no Bethe ansatz solution found")
        energy, mp, I, J = found
        # Hellmann-Feynman D = dE/dU, Richardson-extrapolated central differences on the same
        # Bethe state. Error estimate: the change between steps h and 2h, plus round-off.
        h = FD_STEP * max(1.0, U)
        e = {s: _energy_at(L, N, mp, U + s * h, t, I, J) for s in (-2, -1, 1, 2)}
        central_h = (e[1] - e[-1]) / (2 * h)
        central_2h = (e[2] - e[-2]) / (4 * h)
        d_total = (4 * central_h - central_2h) / 3
        d_error = abs(central_h - central_2h) / 3 + 1e-12 * max(1.0, abs(energy)) / h
        observables = standard_observables(model, energy - model.mu * N, d_total)
        n = model.num_sites
        observables["double_occupancy"] = Observable(d_total / n, d_error / n)
        observables["kinetic_energy"] = Observable(
            observables["kinetic_energy"].value, model.U * d_error
        )
        metadata = {
            "total_spin": (N - 2 * mp) / 2,
            "closed_shell_shortcut": half_filled_closed_shell(L, N, M),
            "quantum_numbers_I": [float(v) for v in I],
            "quantum_numbers_J": [float(v) for v in J],
        }
        return observables, metadata
