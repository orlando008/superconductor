"""Layer 2: many solvers behind one interface.

Planned integrations (wrapping existing open-source codes rather than replacing them):
TRIQS (DMFT / cluster DMFT), ALF (auxiliary-field QMC), ITensor and TeNPy (DMRG / tensor
networks), NetKet (neural-network quantum states), and a diagrammatic Monte Carlo backend.
"""

from superconductor.solvers.base import (
    CostEstimate,
    NotApplicableError,
    Observable,
    Result,
    Solver,
    standard_observables,
)
from superconductor.solvers.bethe import BetheAnsatz
from superconductor.solvers.ed import ExactDiagonalization
from superconductor.solvers.exact_limits import AtomicLimit, FreeFermions

_REGISTRY: dict[str, type[Solver]] = {
    cls.name: cls for cls in (ExactDiagonalization, FreeFermions, AtomicLimit, BetheAnsatz)
}


def register_solver(cls: type[Solver]) -> type[Solver]:
    """Register a solver class under its ``name`` (usable as a decorator)."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Solver name {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_solver(name: str) -> Solver:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"Unknown solver {name!r}; available: {sorted(_REGISTRY)}") from None


def list_solvers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "AtomicLimit",
    "BetheAnsatz",
    "CostEstimate",
    "ExactDiagonalization",
    "FreeFermions",
    "NotApplicableError",
    "Observable",
    "Result",
    "Solver",
    "get_solver",
    "list_solvers",
    "register_solver",
    "standard_observables",
]
