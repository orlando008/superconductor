# superconductor

A correlated-electron workbench: shared infrastructure to help AI agent swarms (and human
groups) work on Grand Challenge #1, **a predictive theory of cuprate superconductivity**.

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

**Status:** layer 1 and a first slice of layer 2 are implemented. See
[docs/roadmap.md](docs/roadmap.md) for what exists, what's next, and how the layers fit together.

## What works today

- `HubbardModel`: a validated, versioned, JSON-serializable t-t'-U Hubbard model spec on a
  square lattice, with a content fingerprint that every result records.
- Three solvers behind one `Solver` interface:
  - `ed`: exact diagonalization (dense, or Lanczos with a check that the degenerate ground
    manifold is complete), ground state and finite temperature
  - `free`: exact solution at U = 0
  - `atomic`: exact solution at t = t' = 0
- Automatic cross-validation (`crosscheck`) across a standard suite of models, plus rigorous
  sanity checks on every result (double-occupancy bounds, free-fermion energy bounds).

The cross-check suite has already paid for itself. It caught ARPACK silently returning the
wrong ground-state energy for a heavily degenerate Hamiltonian (see `solvers/ed.py`).

## Quickstart

```bash
pip install -e ".[dev]"
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
  cli.py        `superconductor` command
examples/       model files
tests/          analytic results, symmetries, cross-checks
docs/           vision and roadmap
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: every result must carry its model
fingerprint, every solver must join the cross-check suite, and tolerances are never loosened
to make a disagreement go away.

## License

MIT
