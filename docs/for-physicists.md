# For physicists

A short brief for researchers in correlated-electron numerics: what this project is, how
results are checked, and where expert judgement is needed. The non-technical overview is
on the [website](https://orlando008.github.io/superconductor/).

## What it is

An open-source platform for **reproducible, cross-checked numerics on the Hubbard model**,
built towards
[Grand Challenge Problem 01: A Predictive Theory of Cuprate Superconductivity](https://grandchallenges.discoveredmaterials.com/cuprate-superconductivity/)
(Discovered Materials). It is an independent community project.

The organizer is not a physicist. Most of the code is written by AI agents, under rules
designed for that situation (see [AGENTS.md](../AGENTS.md)). The project's answer to "why
trust AI-written code?" is that you shouldn't have to: every result is checked by methods
that share no code with the solver that produced it. The most useful thing an expert can
do is try to break it.

## Model and conventions

Single-band t-t′-U Hubbard model on finite square lattices, with periodic or open
boundaries and fixed N↑, N↓ (canonical ensemble):

    H = −t Σ⟨ij⟩σ (c†iσ cjσ + h.c.) − t′ Σ⟨⟨ij⟩⟩σ (c†iσ cjσ + h.c.) + U Σi ni↑ ni↓ − μ N

- t > 0 and t′ < 0 for cuprates. Energies are in units of t and **include −μN**.
- **Periodic direction of length 2:** the wrap-around bond is counted twice, so a periodic
  2×2 cluster has effective hopping 2t. This makes the real-space hopping reproduce
  −2t(cos kx + cos ky) − 4t′ cos kx cos ky exactly on the allowed momenta. Some papers
  treat 2×2 as a 4-site ring without doubling, so take care when comparing.
- At T = 0, observables are averaged uniformly over a degenerate ground manifold (the
  T → 0⁺ limit).
- Every model has a content fingerprint, and every result records it.

Definitions live in exactly one place each: `src/superconductor/model/hubbard.py` and
`lattice.py` for the model, and `solvers/base.py` for observables.

## Solvers

| Solver | What it covers |
| --- | --- |
| `ed` | Exact diagonalization: dense, or Lanczos with a deflation check that the degenerate ground manifold is complete (plain ARPACK silently missed it in one case); finite T from the full spectrum |
| `ed-k` | ED block by block in total momentum. Doped 4×4 clusters are reachable on a 15 GB machine |
| `free` | U = 0, exact at T = 0 and at finite T in the canonical ensemble |
| `atomic` | t = t′ = 0, exact at any U and T |
| `bethe` | Lieb–Wu ground states of periodic 1D rings at any U > 0, including higher-spin ground states of a sector |

## How results are checked

- **Solver against solver:** every applicable pair of solvers is compared on a standard
  suite of models.
- **An independent reference Hamiltonian:** built with Jordan–Wigner operators and
  image-counted hopping, sharing no code with the solvers, and compared through spectra
  and trace moments.
- **Perturbative limits:** the weak-coupling remainder must be ≤ 0 (variational) and scale
  as U²; the strong-coupling remainder at half filling must scale as U⁻³ against an
  independent Heisenberg solver.
- **Mutation tests:** each check is shown to catch a deliberately planted bug.
- **Results registry:** records are append-only and content-addressed. A result counts as
  "reproduced" only when a different person reruns it. Numerical settings, convergence
  diagnostics and hardware are recorded with every result.

As of November 2026: about 1,000 solver comparisons with 0 disagreements, 125 Hamiltonians
matching the independent reference, and 489 tests.

**Literature check.** For 4×4 with 5↑ + 5↓ electrons at U = 4, `ed-k` gives
E/N = −1.22380860 at t′ = 0 and −1.12616014 at t′ = −0.22. Husslein, Fettes & Morgenstern,
[arXiv:cond-mat/9705026](https://arxiv.org/abs/cond-mat/9705026), Tables 1 and 2, give
−1.223808 and −1.126160.

## Where expert judgement is needed

These design questions are blocked on physics decisions. Each issue lists what has to be
settled.

- [#19 Pairing correlations](https://github.com/orlando008/superconductor/issues/19):
  which definitions to use (pair–pair, d-wave form factor, vertex-corrected, pair-binding
  energy), and how to store and compare array-valued observables.
- [#20 Published benchmarks](https://github.com/orlando008/superconductor/issues/20):
  which small-cluster values to trust, and which convention traps to document.
- [#21 Uncertainty and trust levels](https://github.com/orlando008/superconductor/issues/21):
  separating statistical from systematic error, and grading how independent a
  reproduction is.
- [#8 Three-band Emery model](https://github.com/orlando008/superconductor/issues/8):
  schema design and phase conventions.
- [#9 ALF determinant QMC wrapper](https://github.com/orlando008/superconductor/issues/9):
  canonical versus grand-canonical comparison, Trotter extrapolation, error bars.
- [#26 4×4 at half filling](https://github.com/orlando008/superconductor/issues/26):
  whether point group and spin-flip symmetry is the right next step.

Equally valuable: **tell us which questions matter most**, and what in the current design
an expert would distrust.

## What's in it for you

- Reproducible, cross-checked benchmark data and tools your group can use. Result data is
  public domain (CC0), and code is MIT licensed.
- A pool of AI agents to do well-specified implementation work once the physics decisions
  are made.
- Credit: git history, the `contributor` field on every record, `CITATION.cff`, and an
  offer of co-authorship on papers that build on the platform
  (see [GOVERNANCE.md](../GOVERNANCE.md)).

## Ways to help, smallest first

1. **One hour:** comment on one design issue above, or skim this page and the code and
   tell us what you'd distrust.
2. **A few hours a month:** act as a domain reviewer for physics changes to the protected
   core (model format, verification, registry).
3. **Ongoing:** help choose the research questions the platform should answer first.

To get in touch, comment on any issue, or open one with the
"[Test or research idea](https://github.com/orlando008/superconductor/issues/new/choose)"
template.
