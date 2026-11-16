"""Authorship check for pull requests that add result records.

The independence check for reproductions compares ``contributor.handle`` fields. That only
means something if nobody can submit a record under someone else's handle. CI therefore
requires every result record *added* by a pull request to name the PR author as its
contributor.

Exception: a maintainer can apply the ``registry-import`` label to a PR that imports records
in bulk on behalf of others. Only users with triage access or above can apply labels, so
contributors can't exempt themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pydantic

from superconductor.registry.schema import Reproduction, Submission
from superconductor.registry.store import Finding

IMPORT_LABEL = "registry-import"


def _record_type(path: Path) -> type[Submission] | type[Reproduction] | None:
    """Which record type a path holds, from its location; None if it isn't a record."""
    if path.suffix != ".json":
        return None
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "submissions":
        return Submission
    if len(parts) >= 3 and parts[-3] == "reproductions":
        return Reproduction
    return None


def check_added_records_authorship(
    added_paths: Iterable[str | Path], author: str, repo_root: str | Path = "."
) -> list[Finding]:
    """Check that every added result record names ``author`` (a GitHub login).

    ``added_paths`` are paths relative to ``repo_root``, e.g. from
    ``git diff --name-only --diff-filter=A``. Paths that aren't result records (README files,
    ``.gitkeep``) are ignored. GitHub logins are case-insensitive, and so is the comparison.
    """
    if not author or not author.strip():
        raise ValueError("PR author login is empty")
    author = author.strip()
    root = Path(repo_root)
    findings: list[Finding] = []
    for raw in added_paths:
        raw = str(raw).strip()
        if not raw:
            continue
        rel = Path(raw)
        record_type = _record_type(rel)
        if record_type is None:
            continue
        try:
            record = record_type.model_validate_json((root / rel).read_text())
        except (OSError, pydantic.ValidationError) as exc:
            findings.append(Finding("error", raw, "unreadable-record", str(exc).splitlines()[0]))
            continue
        handle = record.contributor.handle
        if handle.lower() != author.lower():
            findings.append(
                Finding(
                    "error",
                    raw,
                    "wrong-author",
                    f"record names contributor {handle!r} but the pull request is by "
                    f"{author!r}. Records must be submitted by the person they name. "
                    f"Maintainers importing records for others can apply the "
                    f"{IMPORT_LABEL!r} label.",
                )
            )
    return findings
