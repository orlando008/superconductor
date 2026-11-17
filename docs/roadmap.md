# Roadmap

The platform has seven layers, each building on the ones below it. The full rationale is
in [vision.md](vision.md).

```
 7  Tools for AI agents      cost estimates · sanity checks · derivation checker · theory registry
 6  Prediction challenge     held-out targets · blind submissions · public scoring   <- highest leverage
 5  Experimental database    Tc(p), T*, gaps, CDW wavevectors, QO frequencies, with provenance
 4  Forward models           ARPES · neutron/RIXS · resistivity/Hall · quantum oscillations
 3  Materials -> models      structure -> DFT -> Wannier -> downfolded (t, t', U, Δ) ± error
 2  Solvers + cross-checks   ED · DQMC/AFQMC · DMRG/PEPS · (c)DMFT · NQS · diagMC
 1  Common model format      lattice · orbitals · hoppings · interactions · filling · T · BCs
```

Status key: ✅ done · 🟡 started · ⬜ not started

## Layer 1: Common model format 🟡

Implemented in `superconductor.model`.

- ✅ Single-band t-t'-U Hubbard model on finite square lattices, periodic/open boundaries
- ✅ Canonical filling (N_up, N_dn), chemical potential, temperature
- ✅ JSON schema via pydantic (strict: unknown fields rejected), schema version field
- ✅ Content fingerprint (`sha256:`) recorded in every result
- ⬜ Three-band Emery model (Cu d_{x²-y²}, O p_x, p_y; charge-transfer energy, U_d, U_p, V_pd)
- ⬜ General lattices / tilted clusters (e.g. Betts clusters), twisted boundary conditions
- ⬜ Longer-range hopping (t''), nearest-neighbour V, Hund's J, spin-orbit, electron-phonon
- ⬜ Grand-canonical ensemble; t-J and Heisenberg limits as derived models
- ⬜ Published JSON Schema file + converters to/from TRIQS, ALF, ITensor, TeNPy, NetKet inputs

## Layer 2: Solvers + automatic cross-validation 🟡

Implemented in `superconductor.solvers` and `superconductor.validation`.

- ✅ `Solver` interface: `applicability()` (with reasons) + `solve()` returning a `Result`
  with named observables, error bars, solver version and model fingerprint
- ✅ Exact diagonalization: dense, Lanczos with degenerate-manifold verification, finite T
- ✅ Exact limits: free fermions (U = 0) and atomic limit (t = 0, any U), both at zero and
  finite temperature (canonical ensemble)
- ✅ Independent reference Hamiltonian (`validation/reference.py`): built with different
  algorithms and no shared code, checked against ED's on 111 models; sanity checks no longer
  run solver code
- ✅ Pairwise cross-validation of all applicable solvers; standard suite of 119 models
  (including finite temperature)
- ⬜ ED with symmetry sectors (momentum, point group, total S) for 16 to 20 site clusters
- ✅ Perturbative limits (`validation/perturbative.py`): weak coupling (remainder ≤ 0 and
  ∝ U²) and strong coupling at half filling (Heisenberg with J = 4t²/U, remainder ∝ U⁻³),
  run by `superconductor crosscheck`
- ✅ Bethe ansatz (Lieb–Wu) solver for periodic 1D chains at any U > 0, up to half filling,
  checked against ED on rings up to 10 sites (including higher-spin and open-shell ground
  states) and reaching chains far beyond ED
- ⬜ More exact references: high-temperature series expansion
- ⬜ Wrappers: ALF (DQMC), TRIQS (DMFT/CDMFT), ITensor/TeNPy (DMRG), NetKet (NQS), diagMC
- ⬜ Observables beyond energy/double occupancy: spin/charge correlations, pairing
  correlations (d-wave), Green's functions, spectral functions
- ⬜ Regime map: for each method pair, the parameter region where agreement is required
- ⬜ Continuous cross-check service with a dashboard of disagreements (bug vs. physics)

## Layer 3: Materials → model parameters ⬜

Reproducible pipeline: structure → DFT → Wannier (Cu-d, O-p) → downfolded parameters with
uncertainty. Run on every known cuprate and nickelate. First target question: Hg-1201
(~97 K, single layer) vs. LSCO (~38 K) on the "same" CuO₂ plane.

## Layer 4: Forward models ⬜

Solver output → what each instrument records: ARPES (with resolution and matrix elements),
neutron and RIXS (S(q, ω)), resistivity and Hall coefficient, quantum-oscillation frequencies.

## Layer 5: Experimental database ⬜

Curated, provenance-tracked cuprate measurements: Tc(p), T*, gap sizes, resistivity slopes,
CDW wavevectors, QO frequencies. Digitize figures from papers; cross-check between sources;
human spot-checks.

## Layer 6: Prediction challenge ⬜

Held-out targets (exact numerics for specific models; unreleased experimental results),
blind submissions from any theory, public scoring with error bars. Layers 1, 2 and 7 already
provide the pieces needed for the first round, which uses ED/DQMC numerics as hidden answers.

## Layer 7: Tools for AI agents 🟡

- 🟡 Automatic sanity checks on every result (`superconductor.validation.sanity`): double
  occupancy bounds, energy consistency, rigorous free-fermion energy bounds.
  Next: sum rules, symmetries, causality/Kramers–Kronig, known limits
- 🟡 Cost estimates before running: `Solver.estimate` and `superconductor estimate`, with
  budget warnings in `solve` and `submit`. Done for ED (calibrated, checked by
  `benchmarks/ed_cost.py`) and the exact limits; next: sign-problem severity and GPU-hours
  for future Monte Carlo and tensor-network solvers
- ⬜ Derivation checker (symbolic algebra, diagram rules, limits)
- ⬜ Theory registry: mechanisms written as executable predictions, scored against layer 6

## Community infrastructure 🟡

- ✅ `AGENTS.md`, `GOVERNANCE.md`, `CONTRIBUTING.md`, issue and PR templates, `CODEOWNERS`
- ✅ Public website explaining the project (`site/`, deployed to GitHub Pages)
- ✅ Results record their numerical settings, convergence diagnostics (e.g. ED residuals and
  Lanczos passes) and hardware; reproductions with different settings get flagged
- ✅ Results registry: append-only, content-addressed submissions and reproductions;
  computed verification status; CI validation with re-runs of cheap results
- ✅ CI check that every added record's `contributor.handle` matches the PR author
  (`superconductor check-authorship`; maintainer override via the `registry-import` label)
- ⬜ Automatic claim expiry and labelling for the task queue
- ⬜ Contribution dashboard (per-person credit, unverified results needing reproduction)
- ⬜ Move the registry to a dedicated append-only store once it outgrows git
- ⬜ Separate scoring service holding hidden answers for layer 6

## Suggested next steps

1. Symmetry-resolved ED to reach 16-site (4×4) clusters, the classic benchmark size.
2. First external solver wrapper (ALF DQMC at half filling, no sign problem), joined to
   the cross-check suite against ED on 2×2 and 4×2.
3. Three-band Emery model in the schema, solvable by ED on a small Cu₂O₄ cluster.
4. A first, small layer-6 round with hidden ED answers, to exercise the scoring pipeline.
