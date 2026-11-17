"""Exact diagonalization resolved by total momentum (translation symmetry).

On a lattice that is periodic in every direction of length > 1, the Hamiltonian commutes
with all G = Lx * Ly lattice translations T_a, so it is block diagonal in total momentum K.
Each block is about G times smaller than the (N_up, N_dn) sector, which cuts memory by about
G. That's what brings 4x4 clusters within reach.

**Basis.** Every occupation-basis state s belongs to an orbit {T_a s}. Its representative r
is the orbit member with the smallest bit key (up bits above down bits). Momentum states are

    |r, K> = (G S_r)^(-1/2) sum_a exp(-i K.a) T_a |r>,

where S_r is the number of translations that map r to itself. A representative is allowed
in block K only if exp(-i K.b) sigma_b(r) = 1 for every such b, where sigma_b is the
fermion sign of the translation. Otherwise the sum cancels to zero.

**Fermion signs.** States are ordered as in ``ed``: up operators by ascending site, then down
operators by ascending site. A translation permutes the sites. Reordering the translated
creation operators back to ascending order gives the sign (-1)^(number of inversions),
counted separately for up and down. The two blocks never mix, so moving one past the other
costs nothing.

**Matrix elements.** If a hop takes r to s with amplitude h (including the hopping sign),
and T_b s = sigma |r'> with r' the representative of s, then

    <r', K| H |r, K> += h * sigma * exp(-i K.b) * sqrt(S_r' / S_r).

The interaction and chemical-potential terms are diagonal and translation-invariant.

**Ground state.** Every block is solved. E0 is the lowest energy over all blocks, and
observables are averaged uniformly over every ground state in every block, the same T -> 0+
convention as ``ed``, so the two must agree exactly. Double occupancy is diagonal in the
representative basis, because it is translation-invariant.
"""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from superconductor.model import HubbardModel
from superconductor.solvers.base import CostEstimate, Observable, Solver, standard_observables
from superconductor.solvers.ed import (
    _DEGENERACY_RTOL,
    _LANCZOS_TOL,
    _RESIDUAL_TOL,
    COST,
    MAX_DEGENERACY,
    estimated_nonzeros,
    sector_dimension,
    species_basis,
)

DENSE_LIMIT = 2048
MAX_STATES = 70_000_000  # full (N_up, N_dn) sector, enumerated once to find orbits
MAX_SITES = 20


# ----- translations, signs and orbits -------------------------------------------------------


def translations(lx: int, ly: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """All translations (ax, ay) and the site permutation each induces (perm[a][i] = T_a(i))."""
    shifts, perms = [], []
    for ay in range(ly):
        for ax in range(lx):
            shifts.append((ax, ay))
            perms.append(
                [((i % lx + ax) % lx) + lx * (((i // lx) + ay) % ly) for i in range(lx * ly)]
            )
    return shifts, np.array(perms)


def _inversion_masks(perm: np.ndarray) -> np.ndarray:
    """masks[i] = bit set of sites j > i with perm[i] > perm[j] (pairs inverted by the map)."""
    n = len(perm)
    masks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                masks[i] |= 1 << j
    return masks


def _permute(bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(bits)
    for i, p in enumerate(perm):
        out |= ((bits >> i) & 1) << int(p)
    return out


def _parity(bits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    inv = np.zeros(bits.shape, dtype=np.int64)
    for i, m in enumerate(masks):
        if m:
            inv += ((bits >> i) & 1) * np.bitwise_count(bits & m).astype(np.int64)
    return inv & 1


class OrbitBasis:
    """Translation orbits of a (N_up, N_dn) sector, and the hopping graph between them."""

    def __init__(self, model: HubbardModel):
        lat = model.lattice
        self.n = n = lat.num_sites
        self.shifts, self.perms = translations(lat.lx, lat.ly)
        self.lx, self.ly = lat.lx, lat.ly
        masks = [_inversion_masks(p) for p in self.perms]
        up_states = species_basis(n, model.filling.n_up).astype(np.int64)
        dn_states = species_basis(n, model.filling.n_dn).astype(np.int64)
        self.n_dn_states = len(dn_states)
        self.up_index = np.full(1 << n, -1, dtype=np.int64)
        self.up_index[up_states] = np.arange(len(up_states))
        self.dn_index = np.full(1 << n, -1, dtype=np.int64)
        self.dn_index[dn_states] = np.arange(len(dn_states))

        # Canonical form of every state: smallest key over all translations.
        up = np.repeat(up_states, len(dn_states))
        dn = np.tile(dn_states, len(up_states))
        own_key = (up << n) | dn
        best_key = own_key.copy()
        best_shift = np.zeros(len(up), dtype=np.int8)
        best_sign = np.ones(len(up), dtype=np.int8)
        for a in range(1, len(self.shifts)):
            tu, td = _permute(up, self.perms[a]), _permute(dn, self.perms[a])
            key = (tu << n) | td
            better = key < best_key
            if better.any():
                sign = 1 - 2 * ((_parity(up, masks[a]) + _parity(dn, masks[a])) & 1)
                best_key[better] = key[better]
                best_shift[better] = a
                best_sign[better] = sign[better].astype(np.int8)
        is_rep = best_key == own_key
        self.rep_up, self.rep_dn = up[is_rep], dn[is_rep]
        rep_keys = own_key[is_rep]
        order = np.argsort(rep_keys)
        self.rep_up, self.rep_dn, rep_keys = self.rep_up[order], self.rep_dn[order], rep_keys[order]
        # For every state: which representative, via which translation, with which sign.
        self.state_rep = np.searchsorted(rep_keys, best_key).astype(np.int64)
        self.state_shift, self.state_sign = best_shift, best_sign
        del up, dn, own_key, best_key

        # Stabilizers of the representatives: translations mapping r to itself, and the sign.
        g = len(self.shifts)
        self.stab = np.zeros((len(rep_keys), g), dtype=bool)
        self.stab_sign = np.ones((len(rep_keys), g), dtype=np.int8)
        for a in range(g):
            tu, td = _permute(self.rep_up, self.perms[a]), _permute(self.rep_dn, self.perms[a])
            fixed = ((tu << n) | td) == rep_keys
            self.stab[:, a] = fixed
            sign = 1 - 2 * ((_parity(self.rep_up, masks[a]) + _parity(self.rep_dn, masks[a])) & 1)
            self.stab_sign[:, a] = sign
        self.stab_size = self.stab.sum(1)
        self.double = np.bitwise_count(self.rep_up & self.rep_dn).astype(float)
        self.diagonal = model.U * self.double - model.mu * model.filling.total
        self._build_hops(model)

    def _state_index(self, up: np.ndarray, dn: np.ndarray) -> np.ndarray:
        return self.up_index[up] * self.n_dn_states + self.dn_index[dn]

    def _build_hops(self, model: HubbardModel) -> None:
        """Every hop from every representative: (from, to, translation, signed amplitude)."""
        h = model.hopping_matrix()
        pairs = [(i, j, h[i, j]) for i, j in zip(*np.nonzero(h), strict=True) if i != j]
        frm, to, shift, amp = [], [], [], []
        reps = np.arange(len(self.rep_up))
        for spin in (0, 1):
            bits = self.rep_up if spin == 0 else self.rep_dn
            for i, j, hij in pairs:
                ok = (((bits >> j) & 1) == 1) & (((bits >> i) & 1) == 0)
                if not ok.any():
                    continue
                b = bits[ok]
                lo, hi = min(i, j), max(i, j)
                between = b & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
                hop_sign = 1 - 2 * (np.bitwise_count(between).astype(np.int64) & 1)
                new = b ^ (1 << j) ^ (1 << i)
                up = new if spin == 0 else self.rep_up[ok]
                dn = self.rep_dn[ok] if spin == 0 else new
                s = self._state_index(up, dn)
                frm.append(reps[ok])
                to.append(self.state_rep[s])
                shift.append(self.state_shift[s])
                amp.append(hij * hop_sign * self.state_sign[s])
        self.hop_from = np.concatenate(frm) if frm else np.zeros(0, dtype=np.int64)
        self.hop_to = np.concatenate(to) if to else np.zeros(0, dtype=np.int64)
        self.hop_shift = np.concatenate(shift) if shift else np.zeros(0, dtype=np.int8)
        self.hop_amp = np.concatenate(amp).astype(float) if amp else np.zeros(0)

    # ----- momentum blocks ----------------------------------------------------------------

    def momenta(self) -> list[tuple[int, int]]:
        """Momentum labels (mx, my): K = 2 pi (mx / Lx, my / Ly)."""
        return [(mx, my) for my in range(self.ly) for mx in range(self.lx)]

    def _phases(self, m: tuple[int, int]) -> np.ndarray:
        """exp(-i K . a) for every translation a."""
        kx, ky = 2 * np.pi * m[0] / self.lx, 2 * np.pi * m[1] / self.ly
        return np.array([np.exp(-1j * (kx * ax + ky * ay)) for ax, ay in self.shifts])

    def block(self, m: tuple[int, int]) -> tuple[sp.csr_matrix, np.ndarray]:
        """Hamiltonian block at momentum m, and the indices of the representatives it uses."""
        ph = self._phases(m)
        # A representative survives if exp(-iK.b) sigma_b = 1 on its whole stabilizer.
        compat = np.where(self.stab, ph[None, :] * self.stab_sign, 1.0)
        allowed = np.all(np.abs(compat - 1.0) < 1e-9, axis=1)
        reps = np.flatnonzero(allowed)
        local = np.full(len(allowed), -1, dtype=np.int64)
        local[reps] = np.arange(len(reps))
        keep = allowed[self.hop_from] & allowed[self.hop_to]
        f, t = local[self.hop_from[keep]], local[self.hop_to[keep]]
        norm = np.sqrt(self.stab_size[self.hop_to[keep]] / self.stab_size[self.hop_from[keep]])
        vals = self.hop_amp[keep] * ph[self.hop_shift[keep]] * norm
        real = np.allclose(ph.imag, 0.0)
        if real:
            vals = vals.real
        dim = len(reps)
        ham = sp.csr_matrix((vals, (t, f)), shape=(dim, dim))
        ham = ham + sp.diags(self.diagonal[reps].astype(ham.dtype))
        return ham.tocsr(), reps


# ----- eigen-solver for a Hermitian block ----------------------------------------------------


def _hermitian_ground_manifold(ham: sp.csr_matrix, dense_limit: int):
    """(E0, ground-manifold basis, Lanczos passes or None) for a real or complex block."""
    dim = ham.shape[0]
    if dim == 0:
        return None
    if dim <= dense_limit:
        energies, vectors = np.linalg.eigh(ham.toarray())
        tol = _DEGENERACY_RTOL * max(1.0, abs(energies[0]))
        ground = np.abs(energies - energies[0]) <= tol
        return float(energies[0]), vectors[:, ground], None
    rng = np.random.default_rng(0)
    radius = float(np.abs(ham).sum(axis=1).max())
    shift = 4.0 * radius + 1.0
    basis = np.zeros((dim, 0), dtype=ham.dtype)
    e0, passes = None, 0
    while basis.shape[1] <= MAX_DEGENERACY:
        passes += 1
        q = basis

        def matvec(v, q=q):
            v = np.ravel(v)
            return ham @ v + shift * (q @ (q.conj().T @ v))

        op = LinearOperator((dim, dim), matvec=matvec, dtype=ham.dtype)
        v0 = rng.standard_normal(dim).astype(ham.dtype)
        vals, vecs = eigsh(op, k=1, which="SA", v0=v0, tol=_LANCZOS_TOL)
        val, vec = float(vals[0].real), vecs[:, 0]
        residual = np.linalg.norm(ham @ vec - val * vec)
        if residual > _RESIDUAL_TOL * max(1.0, abs(val)):
            raise RuntimeError(f"Lanczos did not converge (residual {residual:.2e})")
        tol = _DEGENERACY_RTOL * max(1.0, abs(val if e0 is None else e0))
        if e0 is None:
            e0 = val
        elif val < e0 - tol:
            raise RuntimeError("Lanczos missed the ground state on its first pass")
        elif val > e0 + tol:
            return e0, basis, passes
        vec = vec - basis @ (basis.conj().T @ vec)
        basis = np.column_stack([basis, vec / np.linalg.norm(vec)])
    raise RuntimeError(f"ground-state degeneracy exceeds {MAX_DEGENERACY}")


_SHARED: dict[str, Any] = {}  # basis shared with forked worker processes (copy-on-write)


def _solve_block(m: tuple[int, int]):
    """Solve one momentum block and summarize it; the block matrix is discarded afterwards.

    Returns (m, E0 of block, dimension, sum of <D> over the block's ground manifold,
    number of ground states, max residual), or None for an empty block.
    """
    basis, dense_limit = _SHARED["basis"], _SHARED["dense_limit"]
    ham, reps = basis.block(m)
    found = _hermitian_ground_manifold(ham, dense_limit)
    if found is None:
        return None
    e, vecs, _ = found
    d_sum = float(((np.abs(vecs) ** 2).T @ basis.double[reps]).sum())
    residual = float(np.linalg.norm(ham @ vecs - e * vecs, axis=0).max())
    return m, e, ham.shape[0], d_sum, vecs.shape[1], residual


class MomentumED(Solver):
    """Exact diagonalization block by block in total momentum (periodic lattices).

    ``workers`` > 1 solves momentum blocks in parallel processes (Linux/macOS fork); results
    are identical, only wall time changes. It defaults to the ``SUPERCONDUCTOR_WORKERS``
    environment variable (else 1), so commands such as ``superconductor submit`` can use it.

    Start Python with ``OPENBLAS_NUM_THREADS=1`` (or ``OMP_NUM_THREADS=1``) when using
    workers. Otherwise each worker's BLAS threads compete for the same cores: on a 4-core
    machine, one Lanczos pass on a 4x4 block took 14 s with threaded BLAS and 9.8 s
    single-threaded.
    """

    name = "ed-k"
    version = "0.1.0"

    def __init__(self, dense_limit: int = DENSE_LIMIT, workers: int | None = None):
        self.dense_limit = dense_limit
        if workers is None:  # e.g. for `superconductor submit`, which builds solvers by name
            import os

            workers = int(os.environ.get("SUPERCONDUCTOR_WORKERS", "1"))
        self.workers = max(1, workers)

    def applicability(self, model: HubbardModel) -> str | None:
        lat = model.lattice
        for length, boundary, axis in (
            (lat.lx, lat.boundary_x, "x"),
            (lat.ly, lat.boundary_y, "y"),
        ):
            if length > 1 and boundary != "periodic":
                return f"needs periodic boundaries for translation symmetry ({axis} is open)"
        if lat.num_sites > MAX_SITES:
            return f"more than {MAX_SITES} sites"
        if model.temperature != 0.0:
            return "ground state only"
        dim = sector_dimension(model)
        if dim > MAX_STATES:
            return f"sector of {dim:,} states exceeds the {MAX_STATES:,} that can be enumerated"
        return None

    def settings(self) -> dict[str, Any]:
        return {
            "dense_limit": self.dense_limit,
            "max_states": MAX_STATES,
            "degeneracy_rtol": _DEGENERACY_RTOL,
            "residual_tol": _RESIDUAL_TOL,
            "lanczos_tol": _LANCZOS_TOL,
            "max_degeneracy": MAX_DEGENERACY,
        }  # workers is deliberately absent: it changes wall time, never the result

    def estimate(self, model: HubbardModel) -> CostEstimate:
        """Memory falls by about the number of translations G; total time stays close to ED's,
        because the blocks together have as many non-zeros as the full sector. Enumerating the
        orbits costs about G passes over the full sector."""
        reason = self.applicability(model)
        dim = sector_dimension(model)
        g = model.num_sites
        nnz = estimated_nonzeros(model)
        n = model.num_sites
        enumerate_s = 3e-9 * dim * g * 4 * n
        block_nnz = nnz / g
        seconds = COST["base_seconds"] + enumerate_s + 1.5 * COST["lanczos_seconds_per_nnz"] * nnz
        memory = 40.0 * dim + 30.0 * nnz / g + 3 * COST["sparse_bytes_per_nnz"] * block_nnz
        return CostEstimate(
            solver=self.name,
            feasible=reason is None,
            seconds=seconds,
            memory_bytes=memory,
            driver=f"sector {dim:,} states in {g} momentum blocks of about {dim // g:,}",
            reason=reason,
            details={"sector_dimension": dim, "blocks": g, "block_dimension": dim // g},
        )

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        basis = OrbitBasis(model)
        _SHARED["basis"], _SHARED["dense_limit"] = basis, self.dense_limit
        try:
            momenta = basis.momenta()
            if self.workers > 1:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor

                ctx = mp.get_context("fork")
                with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as pool:
                    blocks = list(pool.map(_solve_block, momenta))
            else:
                blocks = [_solve_block(m) for m in momenta]
        finally:
            _SHARED.clear()
        blocks = [b for b in blocks if b is not None]
        e0 = min(b[1] for b in blocks)
        tol = _DEGENERACY_RTOL * max(1.0, abs(e0))
        ground = [b for b in blocks if b[1] <= e0 + tol]
        count = sum(b[4] for b in ground)
        d_total = sum(b[3] for b in ground) / count
        observables = standard_observables(model, e0, d_total)
        metadata = {
            "sector_dimension": sector_dimension(model),
            "orbits": int(len(basis.rep_up)),
            "block_dimensions": {f"{b[0][0]},{b[0][1]}": int(b[2]) for b in blocks},
            "block_ground_energies": {f"{b[0][0]},{b[0][1]}": b[1] for b in blocks},
            "ground_momenta": [[int(b[0][0]), int(b[0][1])] for b in ground],
            "ground_state_degeneracy": int(count),
            "max_residual": max(b[5] for b in ground),
            "method": "momentum blocks",
        }
        return observables, metadata


def full_sector_dimension(model: HubbardModel) -> int:
    n = model.num_sites
    return comb(n, model.filling.n_up) * comb(n, model.filling.n_dn)
