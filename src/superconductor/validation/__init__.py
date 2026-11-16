"""Verification: cross-validation between solvers and sanity checks on every result."""

from superconductor.validation.crosscheck import (
    Comparison,
    CrossCheckReport,
    compare_results,
    crosscheck,
    run_suite,
    standard_suite,
)
from superconductor.validation.sanity import CheckOutcome, sanity_check

__all__ = [
    "CheckOutcome",
    "Comparison",
    "CrossCheckReport",
    "compare_results",
    "crosscheck",
    "run_suite",
    "sanity_check",
    "standard_suite",
]
