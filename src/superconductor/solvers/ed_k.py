"""Exact diagonalization resolved by symmetry: total momentum, and spin flip at S^z = 0.

On a lattice that is periodic in every direction of length > 1, the Hamiltonian commutes
with all Lx * Ly lattice translations T_a. When N_up = N_dn it also commutes with the global
spin flip F (c_i,up <-> c_i,dn). Together these generate an abelian group of G elements
g = T_a F^f (f = 0 or 1), so the Hamiltonian is block diagonal in sectors (K, p): total
momentum K and spin-flip parity p = +-1 (p is fixed to +1 when spin flip isn't used). Each
block is about G times smaller than the (N_up, N_dn) sector, which brings 4x4 clusters at
half filling within reach of a 15 GB machine.

**Basis.** Every occupation-basis state s belongs to an orbit {g s}. Its representative r
is the orbit member with the smallest bit key (up bits above down bits). Sector states are

    |r; K, p> = (G S_r)^(-1/2) sum_g chi(g) g |r>,    chi(T_a F^f) = exp(-i K.a) p^f,

where S_r is the number of group elements that map r to itself. A representative is allowed
in a sector only if chi(b) sigma_b(r) = 1 for every such b, where sigma_b is the fermion
sign of b acting on r. Otherwise the sum cancels to zero.

**Fermion signs.** States are ordered as in ``ed``: up operators by ascending site, then down
operators by ascending site. A translation permutes the sites. Reordering the translated
creation operators back to ascending order gives the sign (-1)^(number of inversions),
counted separately for the up and the down operators. The spin flip turns |u, d> into
(-1)^(N_up N_dn) |d, u>. That sign is the same for every state of the sector, so it is
absorbed into the definition of F (it relabels p and nothing else); the sign of g = T_a F^f
is then the translation sign of the flipped state. The two blocks of operators are never
interleaved, so the up-block and down-block inversion counts simply add.

**Matrix elements.** If a hop takes r to s with amplitude h (including the hopping sign),
and b s = sigma |r'> with r' the representative of s, then

    <r'; K, p| H |r; K, p> += h * sigma * chi(b) * sqrt(S_r' / S_r).

The interaction and chemical-potential terms are diagonal and symmetric under the group.

**Scale.** For 16 sites each group element's action on one spin species (permuted bits and
inversion parity) is precomputed as a 2^16-entry table, so canonicalizing a state is a
handful of table lookups. The sector is processed in chunks with compact integer types, and
the hopping graph is stored once in CSR form (rows = source representative) with a one-byte
group-element code and a one-byte signed amplitude code per entry. Each sector only adds
its own data array of phases. Representatives not allowed in a sector are masked out of the
Lanczos vectors rather than copied away.

**Ground state.** Every sector is solved. E0 is the lowest energy over all sectors, and
observables are averaged uniformly over every ground state in every sector, the same
T -> 0+ convention as ``ed``, so the two must agree exactly. Double occupancy is diagonal in
the representative basis, because it is symmetric under the whole group.

**Checkpoints.** With a checkpoint directory (``checkpoint_dir`` or the
``SUPERCONDUCTOR_CHECKPOINT_DIR`` environment variable), each finished sector's summary is
written to disk and later runs skip it. Results are identical with or without checkpoints;
they only let runs of several hours survive interruptions.
"""

from __future__ import annotations

import hashlib
import json
import os
from math import comb
from pathlib import Path
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
TINY = 16  # sectors this small always use dense diagonalization
MAX_STATES = 200_000_000  # full (N_up, N_dn) sector, enumerated once (in chunks) to find orbits
MAX_SITES = 20
CHUNK = 4_000_000  # states canonicalized per chunk


# ----- group elements, tables, orbits ---------------------------------------------------------


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


def _species_tables(perm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """For every n-bit occupation x: the permuted bits, and the parity of the reordering."""
    x = np.arange(1 << n, dtype=np.int64)
    out = np.zeros_like(x)
    for i, p in enumerate(perm):
        out |= ((x >> i) & 1) << int(p)
    parity = np.zeros_like(x)
    for i in range(n):
        mask = 0
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                mask |= 1 << j
        if mask:
            parity += ((x >> i) & 1) * np.bitwise_count(x & mask).astype(np.int64)
    key_dtype = np.uint32 if n <= 32 else np.int64
    return out.astype(key_dtype), (parity & 1).astype(np.int8)


class OrbitBasis:
    """Orbits of a (N_up, N_dn) sector under translations (and spin flip), and the hop graph."""

    def __init__(self, model: HubbardModel, spin_flip: bool | None = None, chunk: int = CHUNK):
        lat = model.lattice
        self.n = n = lat.num_sites
        self.lx, self.ly = lat.lx, lat.ly
        nu, nd = model.filling.n_up, model.filling.n_dn
        self.spin_flip = (nu == nd) if spin_flip is None else bool(spin_flip and nu == nd)
        self.shifts, perms = translations(lat.lx, lat.ly)
        # Group elements: (translation index, flip). Element 0 is the identity.
        self.elements = [
            (a, f) for f in ((0, 1) if self.spin_flip else (0,)) for a in range(len(self.shifts))
        ]
        tables = [_species_tables(p, n) for p in perms]
        self.key_bits = n
        up_states = species_basis(n, nu).astype(np.int64)
        dn_states = species_basis(n, nd).astype(np.int64)
        self.n_dn_states = len(dn_states)
        self.up_index = np.full(1 << n, -1, dtype=np.int64)
        self.up_index[up_states] = np.arange(len(up_states))
        self.dn_index = np.full(1 << n, -1, dtype=np.int64)
        self.dn_index[dn_states] = np.arange(len(dn_states))
        total = len(up_states) * len(dn_states)

        # Pass 1: canonical key, group element and sign of every state, chunk by chunk.
        key_t = np.uint64
        best_key = np.empty(total, dtype=key_t)
        self.state_elem = np.empty(total, dtype=np.int8)
        self.state_sign = np.empty(total, dtype=np.int8)
        rows_per_chunk = max(1, chunk // max(1, len(dn_states)))
        rep_keys = []
        for start in range(0, len(up_states), rows_per_chunk):
            ups = up_states[start : start + rows_per_chunk]
            up = np.repeat(ups, len(dn_states))
            dn = np.tile(dn_states, len(ups))
            lo = start * len(dn_states)
            own = (up.astype(key_t) << key_t(n)) | dn.astype(key_t)
            best = own.copy()
            elem = np.zeros(len(up), dtype=np.int8)
            sign = np.ones(len(up), dtype=np.int8)
            for g, (a, f) in enumerate(self.elements):
                if g == 0:
                    continue
                perm_t, par_t = tables[a]
                first, second = (dn, up) if f else (up, dn)
                key = (perm_t[first].astype(key_t) << key_t(n)) | perm_t[second].astype(key_t)
                better = key < best
                if better.any():
                    best[better] = key[better]
                    elem[better] = g
                    s = 1 - 2 * ((par_t[first[better]] + par_t[second[better]]) & 1)
                    sign[better] = s.astype(np.int8)
            best_key[lo : lo + len(up)] = best
            self.state_elem[lo : lo + len(up)] = elem
            self.state_sign[lo : lo + len(up)] = sign
            rep_keys.append(own[best == own])
        rep_keys = np.sort(np.concatenate(rep_keys))
        # Pass 2: which representative each state belongs to.
        self.state_rep = np.empty(total, dtype=np.int32)
        for lo in range(0, total, chunk):
            self.state_rep[lo : lo + chunk] = np.searchsorted(rep_keys, best_key[lo : lo + chunk])
        del best_key
        mask = (1 << n) - 1
        self.rep_up = (rep_keys >> key_t(n)).astype(np.int64)
        self.rep_dn = (rep_keys & key_t(mask)).astype(np.int64)
        self.n_reps = len(rep_keys)

        # Stabilizers: which elements fix each representative, and with which sign.
        n_g = len(self.elements)
        self.stab = np.zeros((self.n_reps, n_g), dtype=bool)
        self.stab_sign = np.ones((self.n_reps, n_g), dtype=np.int8)
        for g, (a, f) in enumerate(self.elements):
            perm_t, par_t = tables[a]
            first, second = (self.rep_dn, self.rep_up) if f else (self.rep_up, self.rep_dn)
            key = (perm_t[first].astype(key_t) << key_t(n)) | perm_t[second].astype(key_t)
            self.stab[:, g] = key == rep_keys
            self.stab_sign[:, g] = (1 - 2 * ((par_t[first] + par_t[second]) & 1)).astype(np.int8)
        self.stab_size = self.stab.sum(1)
        self.double = np.bitwise_count(self.rep_up & self.rep_dn).astype(float)
        self.diagonal = model.U * self.double - model.mu * model.filling.total
        self._build_hops(model)
        # The per-state tables are only needed to build the hop graph.
        del self.state_rep, self.state_elem, self.state_sign

    # ----- hop graph in CSR form (rows = source representative) ----------------------------

    def _hop_terms(self, model: HubbardModel):
        h = model.hopping_matrix()
        values = sorted({float(abs(v)) for v in h[h != 0]})
        self.amp_table = np.array([0.0] + values)  # code 0 unused; sign carried separately
        pairs = []
        for i, j in zip(*np.nonzero(h), strict=True):
            if i != j:
                code = values.index(float(abs(h[i, j]))) + 1
                pairs.append((int(i), int(j), code * (1 if h[i, j] > 0 else -1)))
        return pairs

    def _targets(self, spin, i, j):
        """Representatives that can hop j -> i in this spin, and where they land."""
        bits = self.rep_up if spin == 0 else self.rep_dn
        ok = np.flatnonzero((((bits >> j) & 1) == 1) & (((bits >> i) & 1) == 0))
        b = bits[ok]
        lo, hi = min(i, j), max(i, j)
        between = b & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
        hop_sign = 1 - 2 * (np.bitwise_count(between).astype(np.int64) & 1)
        new = b ^ (1 << j) ^ (1 << i)
        up = new if spin == 0 else self.rep_up[ok]
        dn = self.rep_dn[ok] if spin == 0 else new
        s = self.up_index[up] * self.n_dn_states + self.dn_index[dn]
        return ok, self.state_rep[s], self.state_elem[s], hop_sign * self.state_sign[s]

    def _build_hops(self, model: HubbardModel) -> None:
        pairs = self._hop_terms(model)
        counts = np.zeros(self.n_reps, dtype=np.int64)
        for spin in (0, 1):
            bits = self.rep_up if spin == 0 else self.rep_dn
            for i, j, _ in pairs:
                counts += (((bits >> j) & 1) == 1) & (((bits >> i) & 1) == 0)
        self.indptr = np.zeros(self.n_reps + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])
        nnz = int(self.indptr[-1])
        self.cols = np.empty(nnz, dtype=np.int32)
        self.elem_code = np.empty(nnz, dtype=np.int8)
        self.amp_code = np.empty(nnz, dtype=np.int8)
        cursor = self.indptr[:-1].copy()
        for spin in (0, 1):
            for i, j, code in pairs:
                frm, to, elem, sign = self._targets(spin, i, j)
                pos = cursor[frm]
                self.cols[pos] = to
                self.elem_code[pos] = elem
                self.amp_code[pos] = (code * sign).astype(np.int8)
                cursor[frm] += 1
        # Every sector's matrix shares these arrays. Freeze them: some SciPy operations
        # (taking abs(), slicing, summing) "canonicalize" a CSR matrix in place, sorting and
        # merging entries. On shared arrays that silently scrambled the structure for every
        # later sector. Read-only arrays turn any such attempt into an immediate error.
        for arr in (self.indptr, self.cols, self.elem_code, self.amp_code):
            arr.flags.writeable = False

    # ----- sectors ---------------------------------------------------------------------------

    def sectors(self) -> list[tuple[int, int, int]]:
        """Sector labels (mx, my, p): K = 2 pi (mx / Lx, my / Ly), spin-flip parity p."""
        parities = (1, -1) if self.spin_flip else (1,)
        return [(mx, my, p) for p in parities for my in range(self.ly) for mx in range(self.lx)]

    def momenta(self) -> list[tuple[int, int, int]]:  # kept for callers of the earlier API
        return self.sectors()

    def characters(self, sector: tuple[int, int, int]) -> np.ndarray:
        """chi(g) = exp(-i K . a) p^f for every group element g = T_a F^f."""
        mx, my, p = sector
        kx, ky = 2 * np.pi * mx / self.lx, 2 * np.pi * my / self.ly
        return np.array(
            [
                np.exp(-1j * (kx * self.shifts[a][0] + ky * self.shifts[a][1])) * (p**f)
                for a, f in self.elements
            ]
        )

    def operator(self, sector: tuple[int, int, int]) -> SectorOperator:
        return SectorOperator(self, sector)

    def block(self, sector: tuple[int, int, int]) -> tuple[sp.csr_matrix, np.ndarray]:
        """The sector Hamiltonian as an explicit sparse matrix over its allowed representatives
        (for tests and small systems), and those representatives' indices."""
        op = self.operator(sector)
        return op.explicit(), op.allowed_indices


class SectorOperator:
    """H restricted to one sector, applied matrix-free from the shared hop graph."""

    def __init__(self, basis: OrbitBasis, sector: tuple[int, int, int]):
        self.basis = basis
        chi = basis.characters(sector)
        compat = np.where(basis.stab, chi[None, :] * basis.stab_sign, 1.0)
        self.allowed = np.all(np.abs(compat - 1.0) < 1e-9, axis=1)
        self.allowed_indices = np.flatnonzero(self.allowed)
        self.real = bool(np.allclose(chi.imag, 0.0))
        self.dtype = np.float64 if self.real else np.complex128
        # Entry (row = from, col = to) of H is conj(value(to <- from)) (H is Hermitian).
        # Built in row chunks: the only full-size array is the result. (Building it in one go
        # made five full-size temporaries, about 6 GB for 4x4 at 6+6.)
        amp_table = basis.amp_table
        chi_conj = np.conj(chi).real if self.real else np.conj(chi)
        data = np.empty(len(basis.cols), dtype=self.dtype)
        inv_sqrt_from = 1.0 / np.sqrt(basis.stab_size)
        sqrt_to = np.sqrt(basis.stab_size)
        step = max(1, int(2_000_000 // max(1.0, len(basis.cols) / max(1, basis.n_reps))))
        for r0 in range(0, basis.n_reps, step):
            r1 = min(basis.n_reps, r0 + step)
            lo, hi = basis.indptr[r0], basis.indptr[r1]
            rows = np.repeat(np.arange(r0, r1), np.diff(basis.indptr[r0 : r1 + 1]))
            cols = basis.cols[lo:hi]
            code = basis.amp_code[lo:hi]
            chunk = np.sign(code) * amp_table[np.abs(code)] * sqrt_to[cols] * inv_sqrt_from[rows]
            chunk = chunk * chi_conj[basis.elem_code[lo:hi]]
            chunk[~(self.allowed[rows] & self.allowed[cols])] = 0
            data[lo:hi] = chunk
        self.data = data
        # Only ever used for matrix-vector products, which never modify the matrix.
        self.matrix = sp.csr_matrix(
            (self.data, basis.cols, basis.indptr), shape=(basis.n_reps, basis.n_reps)
        )
        self.diag = basis.diagonal.astype(self.dtype)
        self.dim = int(self.allowed.sum())
        self.shape = (basis.n_reps, basis.n_reps)

    def __matmul__(self, v):
        """Apply H; components on disallowed representatives are kept at zero."""
        out = self.matrix @ v
        if v.ndim == 1:
            out += self.diag * v
            out[~self.allowed] = 0
        else:
            out += self.diag[:, None] * v
            out[~self.allowed, :] = 0
        return out

    def radius(self) -> float:
        """Gershgorin bound on the spectral radius (NumPy only; see the note on freezing)."""
        indptr = self.basis.indptr
        row_sums = np.add.reduceat(np.abs(self.data), indptr[:-1]) if len(self.data) else 0
        row_sums = np.where(np.diff(indptr) > 0, row_sums, 0.0)
        return float(np.max(row_sums + np.abs(self.diag)))

    def explicit(self) -> sp.csr_matrix:
        """This sector as an explicit sparse matrix over its allowed representatives, built
        from copies (for tests and small sectors)."""
        b = self.basis
        rows = np.repeat(np.arange(b.n_reps), np.diff(b.indptr))
        local = np.full(b.n_reps, -1, dtype=np.int64)
        local[self.allowed_indices] = np.arange(self.dim)
        keep = self.allowed[rows] & self.allowed[b.cols]
        m = sp.coo_matrix(
            (self.data[keep], (local[rows[keep]], local[b.cols[keep]])), shape=(self.dim, self.dim)
        ).tocsr()
        return (m + sp.diags(self.diag[self.allowed_indices])).tocsr()


# ----- eigen-solver for one sector -----------------------------------------------------------


def _sector_ground_manifold(op: SectorOperator, dense_limit: int):
    """(E0, ground-manifold basis over the full representative index, Lanczos passes or None)."""
    if op.dim == 0:
        return None
    # ARPACK isn't reliable for tiny problems, so those always go dense.
    if op.dim <= max(dense_limit, TINY):
        energies, vectors = np.linalg.eigh(op.explicit().toarray())
        tol = _DEGENERACY_RTOL * max(1.0, abs(energies[0]))
        ground = np.abs(energies - energies[0]) <= tol
        full = np.zeros((op.shape[0], int(ground.sum())), dtype=vectors.dtype)
        full[op.allowed_indices] = vectors[:, ground]
        return float(energies[0]), full, None
    # Lanczos runs on the allowed representatives only. (Masking them inside the full index
    # space let ARPACK restarts leak into the masked coordinates, which act as spurious
    # zero-energy directions, and the eigenvectors failed their residual check.)
    idx = op.allowed_indices
    n_full, dim = op.shape[0], op.dim

    def apply(small):
        full = np.zeros((n_full,) + small.shape[1:], dtype=op.dtype)
        full[idx] = small
        return (op @ full)[idx]

    rng = np.random.default_rng(0)
    shift = 4.0 * op.radius() + 1.0
    basis = np.zeros((dim, 0), dtype=op.dtype)
    e0, passes = None, 0
    while basis.shape[1] <= MAX_DEGENERACY:
        if basis.shape[1] >= dim - 1:
            # The degenerate manifold (nearly) fills the sector: finish densely on what's
            # left, since a Lanczos pass on 0 or 1 remaining directions can't be trusted.
            energies, vectors = np.linalg.eigh(op.explicit().toarray())
            tol = _DEGENERACY_RTOL * max(1.0, abs(energies[0]))
            ground = np.abs(energies - energies[0]) <= tol
            full = np.zeros((n_full, int(ground.sum())), dtype=op.dtype)
            full[idx] = vectors[:, ground]
            return float(energies[0]), full, passes
        passes += 1
        q = basis

        def matvec(v, q=q):
            v = np.ravel(v)
            return apply(v) + shift * (q @ (q.conj().T @ v))

        lin = LinearOperator((dim, dim), matvec=matvec, dtype=op.dtype)
        v0 = rng.standard_normal(dim).astype(op.dtype)
        vals, vecs = eigsh(lin, k=1, which="SA", v0=v0, tol=_LANCZOS_TOL)
        val, vec = float(vals[0].real), vecs[:, 0]
        residual = np.linalg.norm(apply(vec) - val * vec)
        if residual > _RESIDUAL_TOL * max(1.0, abs(val)):
            raise RuntimeError(f"Lanczos did not converge (residual {residual:.2e})")
        tol = _DEGENERACY_RTOL * max(1.0, abs(val if e0 is None else e0))
        if e0 is None:
            e0 = val
        elif val < e0 - tol:
            raise RuntimeError("Lanczos missed the ground state on its first pass")
        elif val > e0 + tol:
            full = np.zeros((n_full, basis.shape[1]), dtype=op.dtype)
            full[idx] = basis
            return e0, full, passes
        vec = vec - basis @ (basis.conj().T @ vec)
        basis = np.column_stack([basis, vec / np.linalg.norm(vec)])
    raise RuntimeError(f"ground-state degeneracy exceeds {MAX_DEGENERACY}")


def momentum_family(mx: int, my: int, lx: int, ly: int) -> frozenset[tuple[int, int]]:
    """Momenta related to (mx, my) by the lattice's point-group symmetries.

    Reflections x -> -x and y -> -y are symmetries of every periodic Lx x Ly lattice (with
    nearest- and diagonal-neighbour hopping). On a square lattice (Lx = Ly), swapping x and y
    is too, which together with the reflections generates the full C4v. A symmetry R maps the
    block at momentum K onto the block at R K unitarily, so their spectra and ground-state
    observables are identical. The spin-flip parity is unchanged, because spin flip commutes
    with every lattice symmetry.
    """
    family = set()
    for sx in (1, -1):
        for sy in (1, -1):
            a, b = (sx * mx) % lx, (sy * my) % ly
            family.add((a, b))
            if lx == ly:
                family.add((b, a))
    return frozenset(family)


_SHARED: dict[str, Any] = {}  # basis shared with forked worker processes (copy-on-write)


def _solve_sector(sector):
    """Solve one sector and summarize it; its matrix data is discarded afterwards.

    Returns [sector, E0, dimension, sum of <D> over the ground manifold, number of ground
    states, max residual], or None for an empty sector. Writes a checkpoint if configured.
    """
    basis, dense_limit, ckpt = _SHARED["basis"], _SHARED["dense_limit"], _SHARED["checkpoint"]
    op = basis.operator(sector)
    found = _sector_ground_manifold(op, dense_limit)
    if found is None:
        summary = None
    else:
        e, vecs, _ = found
        d_sum = float(((np.abs(vecs) ** 2).T @ basis.double).sum())
        residual = float(np.linalg.norm(op @ vecs - e * vecs, axis=0).max())
        summary = [list(sector), e, op.dim, d_sum, int(vecs.shape[1]), residual]
    if ckpt is not None:
        path = ckpt / f"sector_{sector[0]}_{sector[1]}_{sector[2]}.json"
        path.write_text(json.dumps({"summary": summary}))
    return summary


class MomentumED(Solver):
    """Exact diagonalization block by block in total momentum (and spin-flip parity).

    ``workers`` > 1 solves sectors in parallel processes (Linux/macOS fork); results are
    identical, only wall time changes. It defaults to the ``SUPERCONDUCTOR_WORKERS``
    environment variable (else 1), so commands such as ``superconductor submit`` can use it.

    Start Python with ``OPENBLAS_NUM_THREADS=1`` (or ``OMP_NUM_THREADS=1``) when using
    workers. Otherwise each worker's BLAS threads compete for the same cores: on a 4-core
    machine, one Lanczos pass on a 4x4 block took 14 s with threaded BLAS and 9.8 s
    single-threaded.
    """

    name = "ed-k"
    version = "0.2.0"

    def __init__(
        self,
        dense_limit: int = DENSE_LIMIT,
        workers: int | None = None,
        spin_flip: bool = True,
        checkpoint_dir: str | os.PathLike | None = None,
        point_group: bool = True,
    ):
        self.dense_limit = dense_limit
        # Solve one sector per point-group family and copy its result to the others. It isn't
        # a recorded setting: it changes which sectors are computed, never the results.
        self.point_group = point_group
        if workers is None:  # e.g. for `superconductor submit`, which builds solvers by name
            workers = int(os.environ.get("SUPERCONDUCTOR_WORKERS", "1"))
        self.workers = max(1, workers)
        self.spin_flip = spin_flip
        if checkpoint_dir is None:
            checkpoint_dir = os.environ.get("SUPERCONDUCTOR_CHECKPOINT_DIR") or None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

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
            "spin_flip": self.spin_flip,
            "degeneracy_rtol": _DEGENERACY_RTOL,
            "residual_tol": _RESIDUAL_TOL,
            "lanczos_tol": _LANCZOS_TOL,
            "max_degeneracy": MAX_DEGENERACY,
        }  # workers and checkpoints are deliberately absent: they change wall time, never results

    def _group_size(self, model: HubbardModel) -> int:
        flip = self.spin_flip and model.filling.n_up == model.filling.n_dn
        return model.num_sites * (2 if flip else 1)

    def estimate(self, model: HubbardModel) -> CostEstimate:
        """Memory falls by about the group size G; total time stays close to ED's, because the
        sectors together have as many non-zeros as the full sector. Each parallel worker holds
        its own sector data and Lanczos vectors, so that part scales with ``workers``. (Ignoring
        that once let a 4-worker 4x4 run with t' != 0 exceed 15 GB.)"""
        reason = self.applicability(model)
        dim = sector_dimension(model)
        g = self._group_size(model)
        nnz = estimated_nonzeros(model)
        block_dim, block_nnz = dim / g, nnz / g
        # Enumeration: about 2 table lookups per group element per state.
        seconds = (
            COST["base_seconds"] + 2.5e-9 * dim * g + 1.5 * COST["lanczos_seconds_per_nnz"] * nnz
        )
        shared = 7.0 * dim + 6.0 * block_nnz + g * block_dim + 16.0 * block_dim
        per_worker = 16.0 * block_nnz + 30 * 16.0 * block_dim
        memory = shared + self.workers * per_worker
        return CostEstimate(
            solver=self.name,
            feasible=reason is None,
            seconds=seconds,
            memory_bytes=memory,
            driver=f"sector {dim:,} states in {g} symmetry blocks of about {int(block_dim):,}",
            reason=reason,
            details={"sector_dimension": dim, "blocks": g, "block_dimension": int(block_dim)},
        )

    def _checkpoint_dir(self, model: HubbardModel) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        settings = json.dumps(self.settings(), sort_keys=True)
        tag = hashlib.sha256((model.fingerprint() + self.version + settings).encode()).hexdigest()
        path = self.checkpoint_dir / f"ed-k-{tag[:16]}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        ckpt = self._checkpoint_dir(model)
        basis = OrbitBasis(model, spin_flip=self.spin_flip)
        sectors = basis.sectors()
        done = {}
        if ckpt is not None:
            for s in sectors:
                path = ckpt / f"sector_{s[0]}_{s[1]}_{s[2]}.json"
                if path.exists():
                    done[s] = json.loads(path.read_text())["summary"]
        # Group sectors into point-group families; each family is solved once, preferring a
        # member that's already checkpointed.
        families: dict[tuple, list] = {}
        for s in sectors:
            if self.point_group:
                fam = momentum_family(s[0], s[1], basis.lx, basis.ly)
                key = (min(fam), s[2])
            else:
                key = s
            families.setdefault(key, []).append(s)
        solver_for = {}
        for members in families.values():
            have = [m for m in members if m in done]
            chosen = have[0] if have else members[0]
            for m in members:
                solver_for[m] = chosen
        todo = sorted({c for c in solver_for.values() if c not in done})
        _SHARED.update(basis=basis, dense_limit=self.dense_limit, checkpoint=ckpt)
        try:
            if self.workers > 1 and len(todo) > 1:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor

                ctx = mp.get_context("fork")
                with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as pool:
                    fresh = list(pool.map(_solve_sector, todo))
            else:
                fresh = [_solve_sector(s) for s in todo]
        finally:
            _SHARED.clear()
        solved = dict(done)
        solved.update(zip(todo, fresh, strict=True))
        summaries = []
        for s in sectors:
            source = solved[solver_for[s]]
            if source is not None:
                summaries.append([list(s)] + source[1:])
        e0 = min(b[1] for b in summaries)
        tol = _DEGENERACY_RTOL * max(1.0, abs(e0))
        ground = [b for b in summaries if b[1] <= e0 + tol]
        count = sum(b[4] for b in ground)
        d_total = sum(b[3] for b in ground) / count

        def label(b):
            return ",".join(str(v) for v in b[0])

        observables = standard_observables(model, e0, d_total)
        metadata = {
            "sector_dimension": sector_dimension(model),
            "orbits": int(basis.n_reps),
            "spin_flip": basis.spin_flip,
            "block_dimensions": {label(b): int(b[2]) for b in summaries},
            "block_ground_energies": {label(b): b[1] for b in summaries},
            "ground_momenta": sorted({(int(b[0][0]), int(b[0][1])) for b in ground}),
            "ground_sectors": [list(map(int, b[0])) for b in ground],
            "ground_state_degeneracy": int(count),
            "sectors_solved": len({solver_for[s] for s in sectors}),
            "sectors_inferred_by_point_group": len(sectors) - len({solver_for[s] for s in sectors}),
            "max_residual": max(b[5] for b in ground),
            "method": "symmetry blocks"
            + (" (momentum, spin flip)" if basis.spin_flip else " (momentum)"),
        }
        metadata["ground_momenta"] = [list(m) for m in metadata["ground_momenta"]]
        return observables, metadata


def full_sector_dimension(model: HubbardModel) -> int:
    n = model.num_sites
    return comb(n, model.filling.n_up) * comb(n, model.filling.n_dn)
