# Contributing

This repository is meant to be worked on by many contributors at once, human and AI. The
rules below are what keeps parallel work trustworthy.

- **Using an AI agent?** It must follow [AGENTS.md](AGENTS.md). That file also explains how
  to pick a task, what "done" means, and how to submit or reproduce results.
- **Roles, review, credit, licensing and disputes:** see [GOVERNANCE.md](GOVERNANCE.md).
- **Results:** see [results/README.md](results/README.md).
- **Sign off your commits** (`git commit -s`) to certify the
  [Developer Certificate of Origin](https://developercertificate.org/).

## Ground rules

1. **Every result is traceable.** Solvers return a `Result` carrying the model fingerprint,
   solver name and solver version. Never strip or fake them.
2. **Every solver joins the cross-check suite.** A new solver must declare where it applies
   (`applicability`) and must agree with existing methods wherever their regimes overlap.
   Add models to `validation.standard_suite()` that exercise the overlap.
3. **Disagreements are findings, not noise.** If a cross-check fails, find out whether it's a
   bug or physics. Never loosen a tolerance to make a failure go away without a written
   justification next to the change.
4. **Tests use known answers.** Prefer analytic results, exact symmetries and rigorous bounds
   over regression snapshots. Say in the test where the expected value comes from.
5. **Wrap, don't rewrite.** For established methods (DQMC, DMFT, DMRG, NQS), wrap existing
   open-source codes (ALF, TRIQS, ITensor, TeNPy, NetKet) behind the `Solver` interface.
6. **Results are append-only and need independent reproduction.** Never edit a record under
   `results/`. A result counts once a different contributor has reproduced it.
7. **Conventions live in one place.** Sign conventions, units and indexing are documented in
   `model/hubbard.py` and `model/lattice.py`. Changing them means bumping `SCHEMA_VERSION`.

## Development

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

```bash
ruff check . && ruff format --check .
```

```bash
superconductor crosscheck
```

```bash
superconductor validate --reproduce
```

## Adding a solver

```python
from superconductor.solvers import Solver, register_solver, standard_observables

@register_solver
class MySolver(Solver):
    name = "my-method"
    version = "0.1.0"

    def applicability(self, model):
        return None if model.lattice.num_sites <= 64 else "too many sites"

    def _solve(self, model):
        energy, double_occupancy_total = ...
        return standard_observables(model, energy, double_occupancy_total), {"notes": "..."}
```

Statistical methods must fill in `Observable.error`. Cross-checks use it to widen the
tolerance at 2σ.
