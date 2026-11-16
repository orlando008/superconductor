"""CI authorship check: added result records must name the pull request author."""

import subprocess
import sys

import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.registry import (
    Contributor,
    check_added_records_authorship,
    create_reproduction,
    create_submission,
    write_record,
)

MODEL = HubbardModel(lattice=SquareLattice(lx=2, ly=2), U=4.0, filling=Filling(n_up=2, n_dn=1))


def rel(path, repo):
    return str(path.relative_to(repo))


@pytest.fixture
def repo(tmp_path):
    return tmp_path


def test_own_records_pass(repo):
    sub = create_submission(MODEL, "ed", Contributor(handle="alice"))
    path = write_record(repo / "results", sub)
    assert check_added_records_authorship([rel(path, repo)], "alice", repo) == []


def test_login_comparison_is_case_insensitive(repo):
    """GitHub logins are case-insensitive: 'Alice' and 'alice' are the same account."""
    path = write_record(
        repo / "results", create_submission(MODEL, "ed", Contributor(handle="Alice"))
    )
    assert check_added_records_authorship([rel(path, repo)], "aLiCe", repo) == []


def test_submission_under_someone_elses_handle_fails(repo):
    path = write_record(repo / "results", create_submission(MODEL, "ed", Contributor(handle="bob")))
    findings = check_added_records_authorship([rel(path, repo)], "mallory", repo)
    assert [f.code for f in findings] == ["wrong-author"]
    assert "registry-import" in findings[0].message


def test_forged_reproduction_fails(repo):
    """The attack this check exists for: faking an independent reproduction."""
    sub = create_submission(MODEL, "ed", Contributor(handle="alice"))
    write_record(repo / "results", sub)
    forged = create_reproduction(sub, Contributor(handle="bob"))
    path = write_record(repo / "results", forged)
    findings = check_added_records_authorship([rel(path, repo)], "alice", repo)
    assert [f.code for f in findings] == ["wrong-author"]


def test_non_record_files_are_ignored(repo):
    (repo / "results" / "submissions").mkdir(parents=True)
    (repo / "results" / "submissions" / ".gitkeep").write_text("")
    (repo / "results" / "README.md").write_text("x")
    paths = ["results/submissions/.gitkeep", "results/README.md", "", "  "]
    assert check_added_records_authorship(paths, "alice", repo) == []


def test_unreadable_record_is_reported(repo):
    bad = repo / "results" / "submissions" / "deadbeefdeadbeef.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    findings = check_added_records_authorship([rel(bad, repo)], "alice", repo)
    assert [f.code for f in findings] == ["unreadable-record"]


def test_empty_author_is_an_error(repo):
    with pytest.raises(ValueError):
        check_added_records_authorship([], "  ", repo)


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_ci_pipeline_end_to_end(repo):
    """Mirror the CI step: list files added since the base commit, pipe into the CLI."""
    _git(repo, "init", "-q")
    base_sub = create_submission(MODEL, "ed", Contributor(handle="carol"))
    write_record(repo / "results", base_sub)  # existing record by someone else
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    def run_check(author):
        added = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=A",
                base,
                "HEAD",
                "--",
                "results/submissions",
                "results/reproductions",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "superconductor",
                "check-authorship",
                "--author",
                author,
                "--repo-root",
                str(repo),
            ],
            input=added,
            capture_output=True,
            text=True,
        )

    # PR by alice: reproduces carol's record under her own name. Fine.
    write_record(repo / "results", create_reproduction(base_sub, Contributor(handle="alice")))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "alice reproduces")
    ok = run_check("alice")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "1 added record file(s)" in ok.stdout

    # Same PR also adds a second "independent" reproduction claiming to be dave.
    write_record(repo / "results", create_reproduction(base_sub, Contributor(handle="dave")))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "forged")
    bad = run_check("alice")
    assert bad.returncode == 1
    assert "wrong-author" in bad.stdout and "'dave'" in bad.stdout
    # The pre-existing record by carol is never checked against this PR's author.
    assert "carol" not in bad.stdout
