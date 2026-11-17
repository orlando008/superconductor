"""An independent construction of the Hubbard Hamiltonian, used only for verification.

The solvers build their Hamiltonians from ``HubbardModel.hopping_matrix()``, which comes
from ``SquareLattice.bonds()``. If that shared code were wrong (a sign, a missing or doubled
bond, a boundary error), every comparison between solvers would still agree, because both
sides would inherit the same mistake. This module rebuilds the Hamiltonian from the model's
raw fields with deliberately different algorithms and shares no code with ``solvers/`` or
with the lattice and hopping helpers in ``model/``. A guard test checks that.

Differences from the solver path, on purpose:

1. **Hopping amplitudes by counting images.** For every ordered pair of sites (i, j), list
   the displacement vectors from i to j over all periodic images (only the original one for
   an open boundary, or a direction of length 1). Each image that is a nearest-neighbour
   vector (|dx| + |dy| = 1) contributes -t; each diagonal one (|dx| = |dy| = 1) contributes
   -t'. The solvers instead step forward from each site along fixed directions. Both must
   reproduce the documented convention, including the doubled bond on periodic length-2
   directions (two images, dx = +1 and dx = -1, connect the same pair).
2. **Fermion operators as Jordan-Wigner matrices.** Modes are ordered site by site with the
   spins interleaved (i up, i down, i+1 up, ...), unlike ED's "all up, then all down" bit
   strings. Operators act on the full Fock space (4^N states). The (N_up, N_dn) sector is
   projected out at the end, and fermion signs come from operator products, not bit counting.
3. **Free-fermion reference** for the sanity bounds, from an independent single-particle
   solution of the same image-counted hopping matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, reduce

import numpy as np
import scipy.sparse as sp

from superconductor.model import HubbardModel

MAX_REFERENCE_SITES = 8  # full Fock space is 4^N = 65,536 states at N = 8


def _images(delta: int, length: int, periodic: bool) -> list[int]:
    """Displacements delta + k * length with |value| <= 1, over periodic images."""
    if not periodic or length == 1:
        return [delta] if abs(delta) <= 1 else []
    return [delta + k * length for k in range(-2, 3) if abs(delta + k * length) <= 1]


def reference_hopping(model: HubbardModel) -> np.ndarray:
    """Single-particle matrix h (H_kin = sum_ij,s h_ij c+_is c_js) by counting images."""
    lat = model.lattice
    lx, ly = lat.lx, lat.ly
    px, py = lat.boundary_x == "periodic", lat.boundary_y == "periodic"
    n = lx * ly
    h = np.zeros((n, n))
    for i in range(n):
        xi, yi = i % lx, i // lx
        for j in range(n):
            if i == j:
                continue
            xj, yj = j % lx, j // lx
            for dx in _images(xj - xi, lx, px):
                for dy in _images(yj - yi, ly, py):
                    if abs(dx) + abs(dy) == 1:
                        h[i, j] -= model.t
                    elif abs(dx) == 1 and abs(dy) == 1:
                        h[i, j] -= model.t_prime
    return h


@cache
def _annihilators(num_modes: int) -> tuple[sp.csr_matrix, ...]:
    """Jordan-Wigner annihilation operators c_m on the 2^M-dimensional Fock space."""
    a = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))  # |0><1| on one mode
    z = sp.csr_matrix(np.diag([1.0, -1.0]))  # parity string
    eye = sp.identity(2, format="csr")
    ops = []
    for m in range(num_modes):
        factors = [z] * m + [a] + [eye] * (num_modes - m - 1)
        ops.append(reduce(lambda x, y: sp.kron(x, y, format="csr"), factors))
    return tuple(ops)


@dataclass(frozen=True)
class ReferenceHamiltonian:
    matrix: sp.csr_matrix  # projected onto the (N_up, N_dn) sector
    double_occupancy: sp.csr_matrix  # sum_i n_i,up n_i,dn on the same sector


def reference_hamiltonian(model: HubbardModel) -> ReferenceHamiltonian:
    n = model.num_sites
    if n > MAX_REFERENCE_SITES:
        raise ValueError(f"reference construction limited to {MAX_REFERENCE_SITES} sites")
    h = reference_hopping(model)
    c = _annihilators(2 * n)  # mode 2*i + s, s = 0 (up), 1 (down)
    cd = [op.T.tocsr() for op in c]
    num = [cd[m] @ c[m] for m in range(2 * n)]
    dim = 4**n
    ham = sp.csr_matrix((dim, dim))
    for i in range(n):
        for j in range(n):
            if h[i, j] != 0.0:
                for s in (0, 1):
                    ham = ham + h[i, j] * (cd[2 * i + s] @ c[2 * j + s])
    double = sp.csr_matrix((dim, dim))
    for i in range(n):
        double = double + num[2 * i] @ num[2 * i + 1]
    n_up = sum(num[2 * i] for i in range(n)).diagonal()
    n_dn = sum(num[2 * i + 1] for i in range(n)).diagonal()
    total = n_up + n_dn
    ham = ham + model.U * double - model.mu * sp.diags(total)
    keep = np.flatnonzero(
        np.isclose(n_up, model.filling.n_up) & np.isclose(n_dn, model.filling.n_dn)
    )
    return ReferenceHamiltonian(
        matrix=ham[keep][:, keep].tocsr(), double_occupancy=double[keep][:, keep].tocsr()
    )


def reference_free_ground_state(model: HubbardModel) -> tuple[float, float]:
    """(E0, total double occupancy) of the U = 0 problem, averaged over its ground manifold.

    Independent of the solvers: image-counted hopping, own level filling. A partially filled
    degenerate shell is filled uniformly (each level k/g), which is the uniform average over
    the degenerate many-body ground manifold. The two species are independent, so
    <n_i,up n_i,dn> = <n_i,up> <n_i,dn>.
    """
    levels, vecs = np.linalg.eigh(reference_hopping(model))
    densities, energy = [], -model.mu * model.filling.total
    for count in (model.filling.n_up, model.filling.n_dn):
        occ = np.zeros(len(levels))
        if count:
            fermi = levels[count - 1]
            tol = 1e-9 * max(1.0, abs(fermi))
            full = levels < fermi - tol
            shell = np.abs(levels - fermi) <= tol
            occ[full] = 1.0
            occ[shell] = (count - full.sum()) / shell.sum()
        energy += float(occ @ levels)
        densities.append((vecs**2) @ occ)
    return energy, float(densities[0] @ densities[1])


@dataclass(frozen=True)
class HamiltonianCheck:
    model_name: str | None
    passed: bool
    max_spectrum_difference: float | None  # None when the sector is too big to diagonalize
    moment_differences: tuple[float, ...]  # |Tr H^k (reference) - Tr H^k (solver)|, k = 1..3
    detail: str


def _moments(m: sp.csr_matrix) -> tuple[float, float, float]:
    m2 = m @ m
    return float(m.diagonal().sum()), float(m2.diagonal().sum()), float((m2 @ m).diagonal().sum())


def compare_with(
    model: HubbardModel, solver_matrix: sp.spmatrix, tol: float = 1e-9
) -> HamiltonianCheck:
    """Compare a solver's sector Hamiltonian with the reference, in a basis-independent way.

    The two constructions order their basis states differently, so matrix entries can't be
    compared directly. Instead compare the sorted spectra (when small enough to diagonalize)
    and the trace moments Tr H, Tr H^2, Tr H^3 (always).
    """
    ref = reference_hamiltonian(model).matrix
    sol = sp.csr_matrix(solver_matrix)
    if ref.shape != sol.shape:
        return HamiltonianCheck(
            model.name, False, None, (), f"sector dimensions differ: {ref.shape} vs {sol.shape}"
        )
    scale = max(1.0, float(abs(ref).max()) if ref.nnz else 1.0)
    mref, msol = _moments(ref), _moments(sol)
    diffs = tuple(abs(a - b) for a, b in zip(mref, msol, strict=True))
    ok_moments = all(
        d <= tol * scale**k * max(1, ref.shape[0]) for k, d in enumerate(diffs, start=1)
    )
    spec_diff = None
    if ref.shape[0] <= 2000:
        e_ref = np.linalg.eigvalsh(ref.toarray())
        e_sol = np.linalg.eigvalsh(sol.toarray())
        spec_diff = float(np.max(np.abs(e_ref - e_sol))) if len(e_ref) else 0.0
    ok_spec = spec_diff is None or spec_diff <= tol * scale * 10
    passed = ok_moments and ok_spec
    detail = "match" if passed else f"moment diffs {diffs}, spectrum diff {spec_diff}"
    return HamiltonianCheck(model.name, passed, spec_diff, diffs, detail)
