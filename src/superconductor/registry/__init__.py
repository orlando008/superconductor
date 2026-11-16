"""Results registry: append-only, content-addressed records of crowd-sourced results.

A result is trusted only after independent contributors reproduce it. Statuses are computed
from the records on file, never self-declared. See ``results/README.md``.
"""

from superconductor.registry.authorship import IMPORT_LABEL, check_added_records_authorship
from superconductor.registry.schema import (
    KNOWN_OBSERVABLES,
    SUBMISSION_VERSION,
    Contributor,
    Provenance,
    Reproduction,
    ResultRecord,
    Submission,
    content_id,
)
from superconductor.registry.store import (
    Finding,
    Registry,
    SubmissionStatus,
    create_reproduction,
    create_submission,
    load_registry,
    validate_registry,
    write_record,
)

__all__ = [
    "IMPORT_LABEL",
    "KNOWN_OBSERVABLES",
    "SUBMISSION_VERSION",
    "Contributor",
    "Finding",
    "Provenance",
    "Registry",
    "Reproduction",
    "ResultRecord",
    "Submission",
    "SubmissionStatus",
    "check_added_records_authorship",
    "content_id",
    "create_reproduction",
    "create_submission",
    "load_registry",
    "validate_registry",
    "write_record",
]
