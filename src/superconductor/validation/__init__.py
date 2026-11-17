"""Verification: cross-validation between solvers and sanity checks on every result."""

from superconductor.validation.crosscheck import (
    Comparison,
    CrossCheckReport,
    compare_results,
    crosscheck,
    hamiltonian_suite,
    run_hamiltonian_checks,
    run_suite,
    standard_suite,
)
from superconductor.validation.perturbative import (
    PerturbativeOutcome,
    heisenberg_ground_energy,
    perturbative_suite,
    run_perturbative_checks,
    strong_coupling_check,
    weak_coupling_check,
)
from superconductor.validation.reference import (
    HamiltonianCheck,
    compare_with,
    reference_free_ground_state,
    reference_hamiltonian,
    reference_hopping,
)
from superconductor.validation.sanity import CheckOutcome, sanity_check

__all__ = [
    "CheckOutcome",
    "PerturbativeOutcome",
    "heisenberg_ground_energy",
    "perturbative_suite",
    "run_perturbative_checks",
    "strong_coupling_check",
    "weak_coupling_check",
    "HamiltonianCheck",
    "compare_with",
    "hamiltonian_suite",
    "reference_free_ground_state",
    "reference_hamiltonian",
    "reference_hopping",
    "run_hamiltonian_checks",
    "Comparison",
    "CrossCheckReport",
    "compare_results",
    "crosscheck",
    "run_suite",
    "sanity_check",
    "standard_suite",
]
