"""Loading, validating and creating records in the results registry."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic
import scipy

import superconductor
from superconductor.model import HubbardModel
from superconductor.registry.schema import (
    KNOWN_OBSERVABLES,
    Contributor,
    Provenance,
    Reproduction,
    ResultRecord,
    Submission,
    content_id,
    with_id,
)
from superconductor.solvers import get_solver, list_solvers
from superconductor.solvers.ed import sector_dimension
from superconductor.validation import compare_results, sanity_check

AGREEMENT_ATOL = 1e-8
AGREEMENT_RTOL = 1e-6
CI_REPRODUCE_MAX_DIMENSION = 5000  # CI re-runs ED submissions up to this sector size


@dataclass(frozen=True)
class Finding:
    level: Literal["error", "warning"]
    path: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.path}: [{self.code}] {self.message}"


@dataclass(frozen=True)
class SubmissionStatus:
    state: Literal["unverified", "reproduced", "disputed"]
    agreeing: tuple[str, ...] = ()  # handles of independent contributors who agree
    disagreeing: tuple[str, ...] = ()
    agreeing_other_method: tuple[str, ...] = ()  # agreeing reproductions with another solver


@dataclass
class Registry:
    root: Path
    submissions: dict[str, tuple[Path, Submission]] = field(default_factory=dict)
    reproductions: dict[str, list[tuple[Path, Reproduction]]] = field(default_factory=dict)
    load_findings: list[Finding] = field(default_factory=list)

    def status(self, submission_id: str) -> SubmissionStatus:
        _, sub = self.submissions[submission_id]
        agree, disagree, other = [], [], []
        for _, rep in self.reproductions.get(submission_id, []):
            if same_person(rep.contributor, sub.contributor):
                continue  # not independent; flagged by validation
            if rep.result.model_fingerprint != sub.result.model_fingerprint:
                continue
            if reproduction_agrees(sub, rep):
                agree.append(rep.contributor.handle.lower())
                if rep.provenance.solver != sub.provenance.solver:
                    other.append(rep.contributor.handle.lower())
            else:
                disagree.append(rep.contributor.handle.lower())
        state = "disputed" if disagree else "reproduced" if agree else "unverified"
        return SubmissionStatus(
            state,
            tuple(sorted(set(agree))),
            tuple(sorted(set(disagree))),
            tuple(sorted(set(other))),
        )


def same_person(a: Contributor, b: Contributor) -> bool:
    """GitHub logins are case-insensitive: 'Alice' and 'alice' are one account."""
    return a.handle.lower() == b.handle.lower()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.parent))
    except ValueError:
        return str(path)


def load_registry(root: str | Path = "results") -> Registry:
    root = Path(root)
    reg = Registry(root=root)
    for path in sorted((root / "submissions").glob("*.json")):
        try:
            sub = Submission.model_validate_json(path.read_text())
        except pydantic.ValidationError as exc:
            reg.load_findings.append(Finding("error", _rel(path, root), "invalid-schema", str(exc)))
            continue
        if sub.id in reg.submissions:
            reg.load_findings.append(
                Finding("error", _rel(path, root), "duplicate-id", f"id {sub.id} appears twice")
            )
        reg.submissions[sub.id] = (path, sub)
    for path in sorted((root / "reproductions").glob("*/*.json")):
        try:
            rep = Reproduction.model_validate_json(path.read_text())
        except pydantic.ValidationError as exc:
            reg.load_findings.append(Finding("error", _rel(path, root), "invalid-schema", str(exc)))
            continue
        reg.reproductions.setdefault(rep.submission_id, []).append((path, rep))
    return reg


def reproduction_agrees(sub: Submission, rep: Reproduction) -> bool:
    comparisons = compare_results(
        sub.result.to_result(), rep.result.to_result(), atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL
    )
    return bool(comparisons) and all(c.passed for c in comparisons)


def _check_record_common(record, path: Path, root: Path, expected_name: str) -> list[Finding]:
    out = []
    p = _rel(path, root)
    if content_id(record) != record.id:
        out.append(Finding("error", p, "id-mismatch", "id does not match content (was it edited?)"))
    if path.name != expected_name:
        out.append(Finding("error", p, "bad-filename", f"file must be named {expected_name}"))
    if record.result.solver != record.provenance.solver or (
        record.result.solver_version != record.provenance.solver_version
    ):
        out.append(Finding("error", p, "provenance-mismatch", "result and provenance disagree"))
    if not record.result.observables:
        out.append(Finding("error", p, "no-observables", "result has no observables"))
    unknown = set(record.result.observables) - KNOWN_OBSERVABLES
    if unknown:
        out.append(
            Finding(
                "warning", p, "unknown-observable", f"not cross-checkable yet: {sorted(unknown)}"
            )
        )
    return out


def _cheap_to_rerun(solver_name: str, model: HubbardModel) -> bool:
    if solver_name == "ed":
        return sector_dimension(model) <= CI_REPRODUCE_MAX_DIMENSION
    return solver_name in ("free", "atomic")


def validate_registry(reg: Registry, reproduce: bool = False) -> list[Finding]:
    """Check every record. ``reproduce=True`` also re-runs cheap submissions locally."""
    findings = list(reg.load_findings)
    root = reg.root
    for sid, (path, sub) in reg.submissions.items():
        p = _rel(path, root)
        findings += _check_record_common(sub, path, root, f"{sid}.json")
        if sub.result.model_fingerprint != sub.model.fingerprint():
            findings.append(
                Finding("error", p, "fingerprint-mismatch", "result is for another model")
            )
            continue
        for check in sanity_check(sub.model, sub.result.to_result()):
            if not check.passed:
                findings.append(Finding("error", p, f"sanity:{check.name}", check.detail))
        known = sub.provenance.solver in list_solvers()
        if reproduce and known and _cheap_to_rerun(sub.provenance.solver, sub.model):
            solver = get_solver(sub.provenance.solver)
            if solver.applicability(sub.model) is None:
                fresh = solver.solve(sub.model)
                bad = [
                    c
                    for c in compare_results(
                        sub.result.to_result(), fresh, atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL
                    )
                    if not c.passed
                ]
                for c in bad:
                    findings.append(
                        Finding(
                            "error",
                            p,
                            "ci-reproduction-failed",
                            f"{c.observable}: submitted {c.value_a!r}, rerun {c.value_b!r}",
                        )
                    )
    for sid, reps in reg.reproductions.items():
        for path, rep in reps:
            p = _rel(path, root)
            findings += _check_record_common(rep, path, root, f"{rep.id}.json")
            if path.parent.name != sid:
                findings.append(
                    Finding("error", p, "bad-directory", f"must live in reproductions/{sid}/")
                )
            if sid not in reg.submissions:
                findings.append(Finding("error", p, "unknown-submission", f"no submission {sid}"))
                continue
            _, sub = reg.submissions[sid]
            if same_person(rep.contributor, sub.contributor):
                findings.append(
                    Finding("error", p, "not-independent", "reproducer is the original submitter")
                )
            if rep.result.model_fingerprint != sub.result.model_fingerprint:
                findings.append(
                    Finding("error", p, "fingerprint-mismatch", "reproduction is for another model")
                )
            elif not reproduction_agrees(sub, rep):
                findings.append(
                    Finding("warning", p, "disputed", f"disagrees with submission {sid}")
                )
            if (
                rep.provenance.solver == sub.provenance.solver
                and rep.provenance.solver_settings
                and sub.provenance.solver_settings
                and rep.provenance.solver_settings != sub.provenance.solver_settings
            ):
                changed = sorted(
                    k
                    for k in set(rep.provenance.solver_settings)
                    | set(sub.provenance.solver_settings)
                    if rep.provenance.solver_settings.get(k)
                    != sub.provenance.solver_settings.get(k)
                )
                findings.append(
                    Finding(
                        "warning",
                        p,
                        "different-settings",
                        f"same solver as submission {sid} but different numerical settings: "
                        f"{', '.join(changed)} (fine if deliberate, e.g. a convergence check)",
                    )
                )
    return findings


# ----- creating records ----------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(superconductor.__file__).parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware() -> dict[str, str]:
    """Best-effort description of the machine: CPU model, core count, BLAS library."""
    info = {"machine": platform.machine() or "unknown", "cpu_count": str(os.cpu_count() or "?")}
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info["cpu"] = cpu or "unknown"
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["blas"] = f"{blas.get('name', '?')} {blas.get('version', '')}".strip()
    except Exception:  # noqa: BLE001 - NumPy's config format varies between versions
        info["blas"] = "unknown"
    return info


def current_provenance(
    solver_name: str, solver_version: str, solver_settings: dict | None = None, **extra
) -> Provenance:
    return Provenance(
        package_version=superconductor.__version__,
        git_commit=_git_commit(),
        solver=solver_name,
        solver_version=solver_version,
        solver_settings=dict(solver_settings or {}),
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            **_hardware(),
        },
        **extra,
    )


def create_submission(
    model: HubbardModel, solver_name: str, contributor: Contributor, notes: str | None = None
) -> Submission:
    solver = get_solver(solver_name)
    result = solver.solve(model)
    return with_id(
        Submission,
        created=date.today(),
        contributor=contributor,
        model=model,
        result=ResultRecord.from_result(result),
        provenance=current_provenance(
            solver.name, solver.version, result.metadata.get("solver_settings")
        ),
        notes=notes,
    )


def create_reproduction(
    sub: Submission,
    contributor: Contributor,
    solver_name: str | None = None,
    notes: str | None = None,
) -> Reproduction:
    solver = get_solver(solver_name or sub.provenance.solver)
    result = solver.solve(sub.model)
    return with_id(
        Reproduction,
        submission_id=sub.id,
        created=date.today(),
        contributor=contributor,
        result=ResultRecord.from_result(result),
        provenance=current_provenance(
            solver.name, solver.version, result.metadata.get("solver_settings")
        ),
        notes=notes,
    )


def record_path(root: str | Path, record: Submission | Reproduction) -> Path:
    root = Path(root)
    if isinstance(record, Submission):
        return root / "submissions" / f"{record.id}.json"
    return root / "reproductions" / record.submission_id / f"{record.id}.json"


def write_record(root: str | Path, record: Submission | Reproduction) -> Path:
    path = record_path(root, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2) + "\n")
    return path
