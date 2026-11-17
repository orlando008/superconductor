"""Exact diagonalization: the reference every other method is tested against.

The Hilbert space is restricted to a fixed (N_up, N_dn) sector. Basis states are pairs of
bit strings (one per spin species) with fermion operators ordered all-up-then-all-down, so a
hopping term only picks up a sign from same-species fermions it passes over. The Hamiltonian
is assembled as

    H = T_up (x) 1 + 1 (x) T_dn + U * diag(D) - mu * N

where T_s is the kinetic matrix on one species' basis and D counts doubly occupied sites.
"""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from superconductor.model import HubbardModel
from superconductor.solvers.base import CostEstimate, Observable, Solver, standard_observables

DENSE_LIMIT = 4096  # use dense diagonalization at or below this sector dimension
MAX_DIMENSION = 2_000_000
FINITE_T_LIMIT = 6000  # finite T needs the full spectrum
_DEGENERACY_RTOL = 1e-9
_RESIDUAL_TOL = 1e-7
MAX_DEGENERACY = 256

# Cost model constants, single core. Calibrated with benchmarks/ed_cost.py on a 4-core cloud
# container (OpenBLAS 0.3.34, Haswell kernels, one thread). Re-run that script with --fit on
# other hardware; with --check it verifies predictions are within a factor of 3.
COST = {
    "base_seconds": 0.02,  # imports, setup, small-matrix overheads
    "build_seconds_per_nnz": 4e-8,  # assembling the sparse Hamiltonian
    "dense_seconds_per_dim3": 1.45e-10,  # LAPACK eigh, O(D^3)
    "dense_bytes_per_dim2": 40.0,  # about five D x D float64 arrays at peak
    "lanczos_seconds_per_nnz": 2.0e-6,  # matrix-vector products over all Lanczos passes
    "sparse_bytes_per_nnz": 40.0,  # CSR matrix plus construction temporaries at peak
    "lanczos_bytes_per_dim": 400.0,  # Krylov and deflation vectors
}


def species_basis(num_sites: int, n: int) -> np.ndarray:
    """All bit strings over ``num_sites`` bits with exactly ``n`` bits set, ascending."""
    states = np.arange(1 << num_sites, dtype=np.int64)
    return states[np.bitwise_count(states) == n]


def kinetic_matrix(h: np.ndarray, basis: np.ndarray) -> sp.csr_matrix:
    """Matrix of sum_ij h_ij c+_i c_j on a single-species occupation basis."""
    index = {int(s): k for k, s in enumerate(basis)}
    rows, cols, vals = [], [], []
    pairs = [(i, j, h[i, j]) for i, j in zip(*np.nonzero(h), strict=True) if i != j]
    for col, s in enumerate(basis):
        s = int(s)
        for i, j, amp in pairs:
            if not (s >> j) & 1 or (s >> i) & 1:
                continue
            lo, hi = (i, j) if i < j else (j, i)
            between = s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
            sign = -1.0 if bin(between).count("1") % 2 else 1.0
            rows.append(index[s ^ (1 << j) ^ (1 << i)])
            cols.append(col)
            vals.append(sign * amp)
    dim = len(basis)
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def estimated_nonzeros(model: HubbardModel) -> int:
    """Number of non-zero entries of the sector Hamiltonian, counted without building it.

    Off-diagonal: a hop i <- j of one spin species with n particles on N sites is possible
    from exactly C(N - 2, n - 1) basis states of that species (j occupied, i empty, the other
    n - 1 particles anywhere else). Each such entry appears once for every basis state of
    the other species. Summed over the P directed hopping pairs (i, j) with h_ij != 0:

        P * [C(N-2, n_up-1) * D_dn + C(N-2, n_dn-1) * D_up].

    Diagonal: U * d - mu * (N_up + N_dn) for a state with d doubly occupied sites, which is
    zero (and not stored) for some d; the number of states with each d is known exactly.
    """
    from superconductor.solvers.exact_limits import double_occupancy_degeneracies

    n = model.num_sites
    nu, nd = model.filling.n_up, model.filling.n_dn
    h = model.hopping_matrix()
    pairs = int(np.count_nonzero(h - np.diag(np.diag(h))))

    def hops(k: int) -> int:
        return comb(n - 2, k - 1) if n >= 2 and k >= 1 else 0

    off_diagonal = pairs * (hops(nu) * comb(n, nd) + hops(nd) * comb(n, nu))
    shift = model.mu * (nu + nd)
    diagonal = sum(
        g for d, g in double_occupancy_degeneracies(n, nu, nd).items() if model.U * d - shift != 0.0
    )
    return int(off_diagonal + diagonal)


def sector_dimension(model: HubbardModel) -> int:
    n = model.num_sites
    return comb(n, model.filling.n_up) * comb(n, model.filling.n_dn)


def build_hamiltonian(model: HubbardModel) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return the sector Hamiltonian and the diagonal of the double-occupancy operator."""
    n = model.num_sites
    up = species_basis(n, model.filling.n_up)
    dn = species_basis(n, model.filling.n_dn)
    h = model.hopping_matrix()
    t_up, t_dn = kinetic_matrix(h, up), kinetic_matrix(h, dn)
    eye_up = sp.identity(len(up), format="csr")
    eye_dn = sp.identity(len(dn), format="csr")
    double = np.bitwise_count(up[:, None] & dn[None, :]).ravel().astype(float)
    shift = -model.mu * model.filling.total
    ham = sp.kron(t_up, eye_dn) + sp.kron(eye_up, t_dn)
    ham = ham + sp.diags(model.U * double + shift)
    return ham.tocsr(), double


class ExactDiagonalization(Solver):
    """Exact diagonalization in a fixed (N_up, N_dn) sector.

    ``dense_limit`` sets the sector dimension at or below which the full spectrum is computed
    with dense linear algebra; above it, an iterative sparse path is used.
    """

    name = "ed"
    version = "0.1.0"

    def __init__(self, dense_limit: int = DENSE_LIMIT):
        self.dense_limit = dense_limit

    def applicability(self, model: HubbardModel) -> str | None:
        dim = sector_dimension(model)
        if dim > MAX_DIMENSION:
            return f"sector dimension {dim} exceeds limit {MAX_DIMENSION}"
        if model.temperature > 0 and dim > FINITE_T_LIMIT:
            return f"finite temperature needs full spectrum; dimension {dim} > {FINITE_T_LIMIT}"
        return None

    def estimate(self, model: HubbardModel) -> CostEstimate:
        """Predict single-core time and peak memory from the sector size and sparsity.

        Mirrors the paths ``solve`` takes: a diagonal Hamiltonian (no hopping) is trivial;
        finite temperature and small sectors use dense diagonalization, O(D^3) time and
        O(D^2) memory; larger sectors use Lanczos, roughly linear in the number of non-zeros.
        Lanczos time grows with the ground-state degeneracy (one extra pass per degenerate
        state), which isn't known in advance; the constant assumes the typical case.
        """
        reason = self.applicability(model)
        dim = sector_dimension(model)
        nnz = estimated_nonzeros(model)
        c = COST
        build = c["build_seconds_per_nnz"] * nnz
        sparse_mem = c["sparse_bytes_per_nnz"] * nnz
        diagonal = model.t == 0.0 and model.t_prime == 0.0
        if model.temperature > 0:
            method = "dense (full spectrum, needed for finite temperature)"
            seconds = c["base_seconds"] + build + c["dense_seconds_per_dim3"] * dim**3
            memory = sparse_mem + c["dense_bytes_per_dim2"] * dim**2
        elif diagonal:
            # The ground manifold is stored as dim x degeneracy unit vectors (twice, briefly),
            # and with no hopping its size is known exactly from the double-occupancy count.
            from superconductor.solvers.exact_limits import double_occupancy_degeneracies

            g = double_occupancy_degeneracies(
                model.num_sites, model.filling.n_up, model.filling.n_dn
            )
            if model.U > 0:
                degeneracy = g[min(g)]
            elif model.U < 0:
                degeneracy = g[max(g)]
            else:
                degeneracy = dim
            method = f"diagonal (no hopping; {degeneracy:,} lowest-energy states)"
            seconds = c["base_seconds"] + build + 1e-8 * dim * degeneracy
            memory = sparse_mem + 16.0 * dim * degeneracy
        elif dim <= self.dense_limit:
            method = "dense"
            seconds = c["base_seconds"] + build + c["dense_seconds_per_dim3"] * dim**3
            memory = sparse_mem + c["dense_bytes_per_dim2"] * dim**2
        else:
            method = "Lanczos (sparse)"
            seconds = c["base_seconds"] + build + c["lanczos_seconds_per_nnz"] * nnz
            memory = sparse_mem + c["lanczos_bytes_per_dim"] * dim
        return CostEstimate(
            solver=self.name,
            feasible=reason is None,
            seconds=float(seconds),
            memory_bytes=float(memory),
            driver=f"sector dimension {dim:,}, {method}",
            reason=reason,
            details={"sector_dimension": dim, "estimated_nonzeros": nnz, "method": method},
        )

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        ham, double = build_hamiltonian(model)
        if model.temperature > 0:
            return self._thermal(model, ham, double)

        e0, ground_vectors, gap, method = self._ground_manifold(ham)
        weights = np.abs(ground_vectors) ** 2
        d_total = float(np.mean(weights.T @ double))
        metadata = {
            "sector_dimension": ham.shape[0],
            "ground_state_degeneracy": ground_vectors.shape[1],
            "method": method,
        }
        if gap is not None:
            metadata["gap"] = gap
        return standard_observables(model, e0, d_total), metadata

    def _ground_manifold(self, ham: sp.csr_matrix):
        """Return (E0, orthonormal basis of the full ground manifold, gap or None, method)."""
        dim = ham.shape[0]
        offdiag = ham - sp.diags(ham.diagonal())
        if offdiag.count_nonzero() == 0:
            diag = ham.diagonal()
            e0 = float(diag.min())
            ground = np.abs(diag - e0) <= _tol(e0)
            vecs = _unit_columns(dim, ground)
            rest = diag[~ground]
            return e0, vecs, (float(rest.min() - e0) if rest.size else None), "diagonal"
        if dim <= self.dense_limit:
            energies, vectors = np.linalg.eigh(ham.toarray())
            ground = np.abs(energies - energies[0]) <= _tol(energies[0])
            n0 = int(ground.sum())
            gap = float(energies[n0] - energies[0]) if n0 < dim else None
            return float(energies[0]), vectors[:, ground], gap, "dense"
        return _lanczos_ground_manifold(ham)

    @staticmethod
    def _thermal(model, ham, double):
        energies, vectors = np.linalg.eigh(ham.toarray())
        beta = 1.0 / model.temperature
        boltz = np.exp(-beta * (energies - energies[0]))
        z = boltz.sum()
        p = boltz / z
        energy = float(p @ energies)
        d_total = float(p @ ((np.abs(vectors) ** 2).T @ double))
        free_energy = float(energies[0] - np.log(z) / beta)
        observables = standard_observables(model, energy, d_total)
        observables["free_energy"] = Observable(free_energy)
        observables["entropy"] = Observable(beta * (energy - free_energy))
        return observables, {"sector_dimension": ham.shape[0], "method": "dense-thermal"}


def _tol(energy: float) -> float:
    return _DEGENERACY_RTOL * max(1.0, abs(energy))


def _unit_columns(dim: int, mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    out = np.zeros((dim, idx.size))
    out[idx, np.arange(idx.size)] = 1.0
    return out


def _lanczos_ground_manifold(ham: sp.csr_matrix):
    """Ground manifold via ARPACK Lanczos, with a deflation check for missed degenerate states.

    A single-vector Krylov method finds only one vector per distinct eigenvalue in exact
    arithmetic, so degenerate partners are found (if at all) through round-off. Worse, on
    matrices with very few distinct eigenvalues ARPACK can miss the ground energy entirely:
    the t = 0 cross-check on a 4x2 lattice caught exactly that, which is why diagonal
    Hamiltonians are handled separately. Here we iterate: find the lowest state of H with
    the already-found ground vectors shifted far up, until that lowest state lies above E0.
    """
    dim = ham.shape[0]
    rng = np.random.default_rng(0)
    # Gershgorin bound on the spectral radius; shifting found states by more than the
    # spectral width moves them above everything else.
    radius = float(np.abs(ham).sum(axis=1).max())
    shift = 4.0 * radius + 1.0
    basis = np.zeros((dim, 0))
    e0 = None
    while basis.shape[1] <= MAX_DEGENERACY:
        q = basis

        def matvec(v, q=q):
            v = np.ravel(v)
            return ham @ v + shift * (q @ (q.T @ v))

        op = LinearOperator((dim, dim), matvec=matvec, dtype=float)
        vals, vecs = eigsh(op, k=1, which="SA", v0=rng.standard_normal(dim), tol=1e-13)
        val, vec = float(vals[0]), vecs[:, 0]
        residual = np.linalg.norm(ham @ vec - val * vec)
        if residual > _RESIDUAL_TOL * max(1.0, abs(val)):
            raise RuntimeError(f"Lanczos did not converge (residual {residual:.2e})")
        if e0 is None:
            e0 = val
        elif val < e0 - _tol(e0):
            raise RuntimeError("Lanczos missed the ground state on its first pass")
        elif val > e0 + _tol(e0):
            return e0, basis, val - e0, "lanczos+deflation"
        vec = vec - basis @ (basis.T @ vec)  # keep the basis orthonormal
        basis = np.column_stack([basis, vec / np.linalg.norm(vec)])
    raise RuntimeError(f"ground-state degeneracy exceeds {MAX_DEGENERACY}; use dense ED")
