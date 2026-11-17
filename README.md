# superconductor

A correlated-electron workbench: shared infrastructure to help AI agent swarms (and human
groups) work on [Grand Challenge Problem 01: **A Predictive Theory of Cuprate
Superconductivity**](https://grandchallenges.discoveredmaterials.com/cuprate-superconductivity/), posed by Discovered Materials. This is an independent community
project.

The bet (see [docs/vision.md](docs/vision.md)) is that the bottleneck is *verification*, not
ideas. So instead of hunting directly for a theory, this project builds a simulation and
verification platform where every result is traceable, cross-checked and scorable:

1. **A common model format.** One precise description of the Hamiltonian being solved.
2. **Many solvers behind one interface**, automatically checked against each other.
3. Materials → model parameters (DFT → Wannier → downfolding), with error bars.
4. Forward models of what experiments actually measure (ARPES, RIXS, transport, ...).
5. A machine-readable database of 40 years of cuprate experiments.
6. **A standing prediction challenge**, like CASP for proteins.
7. Tools built for AI agents: cost estimates, sanity checks, derivation checker, theory registry.

**New here?** The [project website](https://orlando008.github.io/superconductor/) explains the
science and the plan in plain language, with interactive visuals (source in [`site/`](site/)).

**Status:** layer 1 and a first slice of layer 2 are implemented. See
[docs/roadmap.md](docs/roadmap.md) for what exists, what's next, and how the layers fit together.

## What works today

- `HubbardModel`: a validated, versioned, JSON-serializable t-t'-U Hubbard model spec on a
  square lattice, with a content fingerprint that every result records.
- Five solvers behind one `Solver` interface:
  - `ed`: exact diagonalization (dense, or Lanczos with a check that the degenerate ground
    manifold is complete), ground state and finite temperature
  - `free`: exact solution at U = 0
  - `atomic`: exact solution at t = t' = 0
  - `bethe`: exact ground states of periodic 1D chains at any U (Lieb–Wu Bethe ansatz)
  - `ed-k`: exact diagonalization block by block in total momentum, for periodic lattices;
    about 16 times less memory than `ed` on 4×4, which brings doped 4×4 clusters within reach
- Automatic cross-validation (`crosscheck`) across a standard suite of models, plus rigorous
  sanity checks on every result (double-occupancy bounds, free-fermion energy bounds).
- A results registry: `submit`, `reproduce`, `validate` and `status` commands for
  crowd-sourced, independently reproduced results.

The cross-check suite has already paid for itself. It caught ARPACK silently returning the
wrong ground-state energy for a heavily degenerate Hamiltonian (see `solvers/ed.py`).

## Quickstart

```bash
pip install -e ".[dev]"
```

```bash
superconductor estimate examples/hubbard_4x2_hole_doped.json
```

```bash
superconductor solve examples/hubbard_4x2_hole_doped.json
```

```bash
superconductor crosscheck
```

```bash
python -m pytest
```

From Python:

```python
from superconductor import Filling, HubbardModel, SquareLattice, get_solver
from superconductor.validation import sanity_check

model = HubbardModel(
    lattice=SquareLattice(lx=4, ly=2),
    t=1.0, t_prime=-0.3, U=8.0,
    filling=Filling(n_up=3, n_dn=3),   # hole doping p = 0.25
)
result = get_solver("ed").solve(model)
print(result["energy_per_site"], result["double_occupancy"], result.model_fingerprint)
print(sanity_check(model, result))
```

## Layout

```
src/superconductor/
  model/        layer 1: lattice + Hamiltonian specification, fingerprints
  solvers/      layer 2: Solver interface, registry, ED and exact-limit solvers
  validation/   cross-validation between solvers; sanity checks on results
  registry/     submission/reproduction records, validation, verification status
  cli.py        `superconductor` command
examples/       model files
results/        crowd-sourced result records (append-only)
tests/          analytic results, symmetries, cross-checks
docs/           vision and roadmap
```

## Contributing (humans and agents)

This repository is meant to be a hub for crowd-sourced work: many people pointing their
agents at it to write code, run calculations, reproduce each other's results, and propose
tests. To make that trustworthy at scale:

- **Pick work** from issues labelled `agent-ready`. Each one has checkable acceptance criteria.
- **Submit results** with `superconductor submit` and **verify others'** with
  `superconductor reproduce`. Records in [`results/`](results/README.md) are append-only and
  content-addressed. A result counts once a *different* contributor reproduces it.
- **Report disagreements** as findings. Never adjust numbers or tolerances to hide them.
- CI checks every PR: tests, the cross-check suite, registry validation (with re-runs of cheap
  results), and that no existing record was changed.

Start with [AGENTS.md](AGENTS.md) (rules for agents and their operators),
[CONTRIBUTING.md](CONTRIBUTING.md), and [GOVERNANCE.md](GOVERNANCE.md) (roles, review, credit,
licensing, disputes).

## License

MIT
