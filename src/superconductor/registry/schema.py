"""File formats for crowd-sourced results.

Two record types, both append-only JSON files:

- :class:`Submission` (``results/submissions/<id>.json``): one result for one model, with
  full provenance and the contributor responsible for it.
- :class:`Reproduction` (``results/reproductions/<submission_id>/<id>.json``): an independent
  re-run of a submission by a different contributor.

Reproductions live in their own files so that thousands of contributors never edit each
other's records (and never hit merge conflicts). A record's ``id`` is a hash of its content,
so a record can't be changed after the fact without its id changing too.

A submission's status (unverified / reproduced / disputed) is never written by the
submitter. It's computed from the reproductions on file; see ``registry.validate``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from superconductor.model import HubbardModel
from superconductor.solvers import Observable, Result

SUBMISSION_VERSION = "0.1"

# Shared observable vocabulary. Unknown names are allowed (with a warning) so new
# observables can be proposed, but they won't be cross-checked until added here.
KNOWN_OBSERVABLES = {
    "energy",
    "energy_per_site",
    "kinetic_energy",
    "double_occupancy",
    "free_energy",
    "entropy",
}

_HANDLE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")  # GitHub username rules


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ObservableRecord(_Strict):
    value: float
    error: float = Field(default=0.0, ge=0.0)


class Contributor(_Strict):
    """Who is responsible for a record. Always a human GitHub handle, even if an agent did it."""

    handle: str = Field(description="GitHub username of the responsible human.")
    agent: str | None = Field(default=None, description="AI agent/model used, if any.")

    @field_validator("handle")
    @classmethod
    def _valid_handle(cls, v: str) -> str:
        if not _HANDLE.match(v):
            raise ValueError(f"{v!r} is not a valid GitHub username")
        return v


class Provenance(_Strict):
    """Everything needed to rerun the calculation."""

    package_version: str
    git_commit: str | None = Field(default=None, description="Commit of this repo that ran.")
    solver: str
    solver_version: str
    solver_settings: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    environment: dict[str, str] = Field(
        default_factory=dict, description="Versions of python, numpy, external codes, ..."
    )
    cpu_core_hours: float | None = Field(default=None, ge=0)
    gpu_hours: float | None = Field(default=None, ge=0)


class ResultRecord(_Strict):
    solver: str
    solver_version: str
    model_fingerprint: str
    observables: dict[str, ObservableRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: Result) -> ResultRecord:
        return cls.model_validate(json.loads(result.to_json()))

    def to_result(self) -> Result:
        return Result(
            solver=self.solver,
            solver_version=self.solver_version,
            model_fingerprint=self.model_fingerprint,
            observables={k: Observable(o.value, o.error) for k, o in self.observables.items()},
            metadata=dict(self.metadata),
        )


def content_id(record: BaseModel) -> str:
    """Hash of a record's content, excluding its own ``id`` field."""
    data = record.model_dump(mode="json", exclude={"id"})
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Submission(_Strict):
    kind: Literal["submission"] = "submission"
    submission_version: Literal["0.1"] = SUBMISSION_VERSION
    id: str
    created: date
    contributor: Contributor
    model: HubbardModel
    result: ResultRecord
    provenance: Provenance
    notes: str | None = None


class Reproduction(_Strict):
    kind: Literal["reproduction"] = "reproduction"
    submission_version: Literal["0.1"] = SUBMISSION_VERSION
    id: str
    submission_id: str
    created: date
    contributor: Contributor
    result: ResultRecord
    provenance: Provenance
    notes: str | None = None


def with_id(record_cls: type[BaseModel], **fields) -> Any:
    """Build a record and fill in its content-derived id."""
    draft = record_cls(id="pending", **fields)
    return record_cls(**{**fields, "id": content_id(draft)})
