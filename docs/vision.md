# A Correlated-Electron Workbench for Cuprate Research

Nov 15, 2026 · @Nick

## Overview

A 100k-agent AI swarm is better spent building a shared simulation and verification platform for strongly correlated electrons than hunting directly for a theory of cuprate superconductivity. Infrastructure work is well-specified engineering with checkable results: code passes tests, independent methods agree where they overlap, outputs match known exact answers. Swarms are good at exactly that, and it attacks the real bottleneck, which is verification.

The platform has seven layers, each building on the ones below it.

See [roadmap.md](roadmap.md) for the seven layers and their status.

The prediction challenge (layer 6) is highlighted because it turns every other layer into a scoreboard that humans and models can be measured against.

## 1. A common language for the problem

Every result on the platform starts from one precise, shared description of the model being solved. Today groups write Hamiltonians, lattices, conventions and units differently, so results are hard to compare.

- One format covers lattice, orbitals, hopping terms, interactions, doping, temperature and boundary conditions.
- Every solver reads it, and every result records exactly which model produced it.
- It's not glamorous, but every other layer depends on it.

## 2. Many solvers behind one interface, checked against each other

No single numerical method covers the whole cuprate phase diagram, and each one fails in a different way. So the platform puts all the major methods behind the common format and makes them check each other.

| Method | Best at | Main limitation |
| --- | --- | --- |
| Exact diagonalization | Exact answers to test everything else against | Only very small clusters |
| Determinant / auxiliary-field quantum Monte Carlo | Unbiased finite-temperature results | Sign problem away from half filling |
| DMRG and 2D tensor networks (PEPS) | Ground states on strips and cylinders | Extrapolating to true 2D |
| DMFT and cluster DMFT | Local correlations, real-material spectra | Misses longer-range correlations |
| Neural-network quantum states | Fast-improving variational accuracy | Optimization is hard to verify |
| Diagrammatic Monte Carlo | Controlled expansions, thermodynamic limit | Convergence at strong coupling |

The key feature is automatic cross-validation. Wherever two methods should agree (small systems, high temperature, weak coupling), the platform checks continuously and flags disagreements as either a bug or real physics. This exhaustive, tedious testing is something humans rarely have time for and a swarm does well.

Build on existing open-source codes rather than replacing them: [TRIQS](https://triqs.github.io/), [ALF](https://alf.physik.uni-wuerzburg.de/), [ITensor](https://itensor.org/), [TeNPy](https://github.com/tenpy/tenpy) and [NetKet](https://www.netket.org/). A big part of the job is connecting them.

## 3. From real materials to model parameters, automatically

A reproducible pipeline turns any crystal structure into model parameters with error bars:

1. Crystal structure in.
2. DFT calculation of the electronic bands.
3. Wannier orbitals for the copper and oxygen states.
4. A downfolded model with its parameters (t, t′, U, charge-transfer energy and so on).
5. Parameters stored with uncertainty estimates.

Run it on every known cuprate and nickelate. Then a question like "why does the mercury cuprate (\~133 K) beat LSCO (\~38 K) on essentially the same CuO₂ plane?" becomes something you can query and compute instead of argue about.

## 4. Predicting what experiments actually measure

Solvers output Green's functions and correlation functions, but experiments measure something else. A forward-model layer converts solver output into the exact quantity each instrument records, including resolution and matrix-element effects:

- Photoemission (ARPES) spectra
- Neutron scattering and RIXS
- Resistivity and Hall coefficient
- Quantum oscillation frequencies

Only then is comparing theory to data a fair comparison.

## 5. A machine-readable record of 40 years of experiments

A curated database of cuprate measurements across all compounds, each value carrying its provenance: Tc against doping, the pseudogap temperature T\*, gap sizes, resistivity slopes, charge-density-wave wavevectors, quantum oscillation frequencies and more.

Much of this exists only as figures in PDFs. Digitizing, cross-checking and reconciling it is massively parallel work, well suited to a swarm, with human spot-checks.

## 6. A standing prediction challenge, like CASP for proteins

This is the highest-leverage layer. CASP drove protein-structure prediction for decades and became the scoreboard AlphaFold won on. The equivalent here:

- Held-out targets: exact numerical results for specific models, plus experimental results not yet released.
- Any theory, from a human group or an AI, submits predictions before the answers are revealed.
- Public scoring with error bars.

This turns "is my theory right?" from a matter of opinion into a measurable benchmark, which is exactly what future LLMs need to train and iterate against.

## 7. Tools built for AI agents

On top of everything, an interface designed so AI agents can use the platform safely and efficiently:

- **Cost estimates before running.** For example: "this calculation will take \~4,000 GPU-hours and is sign-problem limited above p = 0.12."
- **Automatic sanity checks** on every result: sum rules, symmetries, known limits, causality.
- **A derivation checker** that verifies analytic steps (symbolic algebra, diagram rules, limits), so long calculations can't silently go wrong. This targets the error-compounding problem directly.
- **A theory registry** where each proposed mechanism is written as executable predictions instead of prose, so it can be scored automatically against layer 6.

## Why this beats the solve-it-directly swarm

A failed theory hunt leaves almost nothing behind. This platform is valuable whatever happens:

- Human research groups can use it immediately.
- The next generation of models starts from much stronger footing.
- It carries over to nickelates, heavy-fermion materials, moiré materials and other strongly correlated systems.

## Caveats

- **Subtle correctness.** Numerical physics codes can pass every test and still be wrong in ways only an expert notices, such as a biased estimator or a finite-size effect presented as physics. A small core of human domain experts should review the critical parts.
- **Compute for running it.** The platform makes simulations easier, not cheaper. The sign problem is still there; it just gets measured honestly.
- **Upkeep and adoption.** Many scientific software projects die from lack of maintenance or because the community never adopts them. Building on existing codes and communities matters more than technical elegance.
