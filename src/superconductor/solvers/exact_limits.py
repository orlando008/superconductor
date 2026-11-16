"""Exactly solvable limits used as independent references for cross-validation.

These are not general-purpose solvers. They exist so that every general method has known,
independently derived answers to agree with in the regimes where the problem is trivial.
"""

from __future__ import annotations

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


class AtomicLimit(Solver):
    """t = t' = 0: isolated sites; doubly occupied sites only when forced by filling."""

    name = "atomic"

    def applicability(self, model: HubbardModel) -> str | None:
        if model.t != 0.0 or model.t_prime != 0.0:
            return "requires t = t' = 0"
        if model.temperature != 0.0:
            return "finite temperature not implemented yet"
        if model.U < 0:
            return "U < 0 not implemented yet"
        return None

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        n_double = max(0, model.filling.total - model.num_sites)
        energy = model.U * n_double - model.mu * model.filling.total
        return standard_observables(model, float(energy), float(n_double)), {}
