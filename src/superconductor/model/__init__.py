"""Layer 1: a common language for the problem.

Every result on the platform starts from one of these model specifications, and every result
records the specification's fingerprint so it can be traced back to exactly what was solved.
"""

from superconductor.model.hubbard import SCHEMA_VERSION, Filling, HubbardModel, load_model
from superconductor.model.lattice import Boundary, SquareLattice

__all__ = [
    "SCHEMA_VERSION",
    "Boundary",
    "Filling",
    "HubbardModel",
    "SquareLattice",
    "load_model",
]
