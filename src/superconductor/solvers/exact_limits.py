"""Exactly solvable limits used as independent references for cross-validation.

These are not general-purpose solvers. They exist so that every general method has known,
independently derived answers to agree with in the regimes where the problem is trivial.
"""

from __future__ import annotations

import math
from math import comb
from typing import Any

import numpy as np

from superconductor.model import HubbardModel
from superconductor.solvers.base import Observable, Solver, standard_observables

_SHELL_TOL = 1e-9


def ground_occupations(eps: np.ndarray, n: int) -> np.ndarray:
    """T -> 0+ occupation of single-particle levels for ``n`` fermions.

    A partially filled degenerate shell at the Fermi level is filled uniformly (each level
    gets k/g), which is the uniform average over the degenerate many-body ground manifold.
    """
    w = np.zeros_like(eps)
    if n == 0:
        return w
    e_f = eps[n - 1]
    tol = _SHELL_TOL * max(1.0, abs(e_f))
    below = eps < e_f - tol
    shell = np.abs(eps - e_f) <= tol
    w[below] = 1.0
    w[shell] = (n - below.sum()) / shell.sum()
    return w


class FreeFermions(Solver):
    """U = 0: diagonalize the single-particle hopping matrix and fill levels."""

    name = "free"

    def applicability(self, model: HubbardModel) -> str | None:
        if model.U != 0.0:
            return "requires U = 0"
        if model.temperature != 0.0:
            return "canonical finite-temperature free fermions not implemented yet"
        return None

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        eps, phi = np.linalg.eigh(model.hopping_matrix())
        energy = -model.mu * model.filling.total
        densities = []
        open_shell = False
        for n in (model.filling.n_up, model.filling.n_dn):
            w = ground_occupations(eps, n)
            open_shell |= bool(np.any((w > 0) & (w < 1)))
            energy += float(w @ eps)
            densities.append((phi**2) @ w)  # <n_i> for this species
        d_total = float(densities[0] @ densities[1])
        metadata = {"single_particle_levels": eps.tolist(), "open_shell": open_shell}
        return standard_observables(model, energy, d_total), metadata


def double_occupancy_degeneracies(num_sites: int, n_up: int, n_dn: int) -> dict[int, int]:
    """Number of (N_up, N_dn) occupation configurations with exactly ``d`` doubly occupied sites.

    Derivation: choose the d doubly occupied sites, C(N, d). Place the remaining N_up - d up
    electrons on the N - d other sites, C(N - d, N_up - d). The remaining N_dn - d down
    electrons must avoid both the doubles and the singly occupied up sites, which leaves
    N - d - (N_up - d) = N - N_up sites: C(N - N_up, N_dn - d). So

        g(d) = C(N, d) * C(N - d, N_up - d) * C(N - N_up, N_dn - d),

    for max(0, N_up + N_dn - N) <= d <= min(N_up, N_dn). Summing over d gives
    C(N, N_up) * C(N, N_dn), the full sector dimension (a Vandermonde identity), which the
    tests check along with brute-force enumeration.
    """
    n = num_sites
    d_min, d_max = max(0, n_up + n_dn - n), min(n_up, n_dn)
    return {
        d: comb(n, d) * comb(n - d, n_up - d) * comb(n - n_up, n_dn - d)
        for d in range(d_min, d_max + 1)
    }


class AtomicLimit(Solver):
    """t = t' = 0: isolated sites. Exact at any U and temperature (canonical ensemble).

    With no hopping, H = U * D - mu * N is diagonal in the occupation basis, and every state
    with d doubly occupied sites has energy E(d) = U d - mu (N_up + N_dn). All thermodynamics
    follows from the degeneracies g(d) (see ``double_occupancy_degeneracies``):

        Z = sum_d g(d) exp(-E(d) / T),   p(d) = g(d) exp(-E(d) / T) / Z,
        <D> = sum_d p(d) d,   E = U <D> - mu N,   F = -T ln Z,   S = (E - F) / T.

    At T = 0, observables are the uniform average over the ground manifold (the T -> 0+
    limit). For U > 0 that is the minimum d; for U < 0 the maximum d; for U = 0 every state
    is degenerate and <D> is the g(d)-weighted mean.
    """

    name = "atomic"
    version = "0.2.0"

    def applicability(self, model: HubbardModel) -> str | None:
        if model.t != 0.0 or model.t_prime != 0.0:
            return "requires t = t' = 0"
        return None

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        f = model.filling
        g = double_occupancy_degeneracies(model.num_sites, f.n_up, f.n_dn)
        ds = np.array(sorted(g), dtype=float)
        energies = model.U * ds - model.mu * f.total
        log_g = np.array([math.log(g[d]) for d in sorted(g)])
        e_ref = float(energies.min())
        metadata: dict[str, Any] = {"degeneracy_by_double_count": {str(d): g[d] for d in g}}

        if model.temperature == 0.0:
            ground = energies == e_ref  # exact: E(d) is linear in d with an exact float U
            weights = np.exp(log_g[ground] - log_g[ground].max())
            d_total = float(weights @ ds[ground] / weights.sum())
            metadata["ground_state_degeneracy"] = int(sum(g[int(d)] for d in ds[ground]))
            return standard_observables(model, e_ref, d_total), metadata

        beta = 1.0 / model.temperature
        log_w = log_g - beta * (energies - e_ref)  # shifted for numerical stability
        log_z_shifted = float(np.logaddexp.reduce(log_w))
        p = np.exp(log_w - log_z_shifted)
        d_total = float(p @ ds)
        energy = float(p @ energies)
        free_energy = e_ref - log_z_shifted / beta
        observables = standard_observables(model, energy, d_total)
        observables["free_energy"] = Observable(free_energy)
        observables["entropy"] = Observable(beta * (energy - free_energy))
        return observables, metadata
