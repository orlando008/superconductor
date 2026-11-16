"""Correlated-electron workbench for cuprate research.

Layer 1 (``superconductor.model``): a single, precise description of the model being solved.
Layer 2 (``superconductor.solvers`` + ``superconductor.validation``): solvers behind one
interface, automatically checked against each other and against exact limits.

See ``docs/roadmap.md`` for the full seven-layer plan.
"""

from superconductor.model import Filling, HubbardModel, SquareLattice
from superconductor.solvers import Observable, Result, Solver, get_solver, list_solvers

__version__ = "0.1.0"

__all__ = [
    "Filling",
    "HubbardModel",
    "Observable",
    "Result",
    "Solver",
    "SquareLattice",
    "get_solver",
    "list_solvers",
]
