import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.cli import main
from superconductor.registry import (
    Contributor,
    content_id,
    create_reproduction,
    create_submission,
    load_registry,
    validate_registry,
    write_record,
)

MODEL = HubbardModel(
    name="2x2 U=4",
    lattice=SquareLattice(lx=2, ly=2),
    t_prime=-0.3,
    U=4.0,
    filling=Filling(n_up=2, n_dn=1),
)
ALICE, BOB, CAROL = (Contributor(handle=h) for h in ("alice", "bob", "carol"))


@pytest.fixture
def root(tmp_path):
    return tmp_path / "results"


def codes(findings):
    return sorted(f.code for f in findings if f.level == "error")


def test_valid_registry_and_status_lifecycle(root):
    sub = create_submission(MODEL, "ed", ALICE)
    write_record(root, sub)
    reg = load_registry(root)
    assert validate_registry(reg, reproduce=True) == []
    assert reg.status(sub.id).state == "unverified"

    write_record(root, create_reproduction(sub, BOB))
    reg = load_registry(root)
    assert reg.status(sub.id).state == "reproduced"
    assert reg.status(sub.id).agreeing == ("bob",)
    assert validate_registry(reg) == []


def test_reproduction_with_a_different_method_is_tracked(root):
    free_model = MODEL.with_updates(U=0.0)
    sub = create_submission(free_model, "ed", ALICE)
    write_record(root, sub)
    write_record(root, create_reproduction(sub, BOB, solver_name="free"))
    status = load_registry(root).status(sub.id)
    assert status.state == "reproduced" and status.agreeing_other_method == ("bob",)


def _tamper(path: Path, value: float, fix_id: bool):
    data = json.loads(path.read_text())
    data["result"]["observables"]["energy"]["value"] = value
    data["result"]["observables"]["energy_per_site"]["value"] = value / 4
    if fix_id:
        from superconductor.registry import Submission

        data["id"] = "pending"
        data["id"] = content_id(Submission.model_validate(data))
        path.unlink()
        path = path.with_name(f"{data['id']}.json")
    path.write_text(json.dumps(data))
    return path


def test_editing_a_record_breaks_its_id(root):
    sub = create_submission(MODEL, "ed", ALICE)
    path = write_record(root, sub)
    _tamper(path, sub.result.observables["energy"].value + 1e-3, fix_id=False)
    assert "id-mismatch" in codes(validate_registry(load_registry(root)))


def test_plausible_fake_value_caught_by_ci_rerun(root):
    """A consistent-looking record with a wrong (but physically allowed) energy."""
    sub = create_submission(MODEL, "ed", ALICE)
    _tamper(write_record(root, sub), sub.result.observables["energy"].value + 1e-3, fix_id=True)
    reg = load_registry(root)
    assert codes(validate_registry(reg)) == []  # passes static checks...
    assert "ci-reproduction-failed" in codes(validate_registry(reg, reproduce=True))  # ...not rerun


def test_unphysical_value_caught_by_sanity_checks(root):
    sub = create_submission(MODEL, "ed", ALICE)
    _tamper(write_record(root, sub), -1000.0, fix_id=True)
    assert "sanity:energy_free_fermion_bounds" in codes(validate_registry(load_registry(root)))


def test_self_reproduction_is_not_independent(root):
    sub = create_submission(MODEL, "ed", ALICE)
    write_record(root, sub)
    write_record(root, create_reproduction(sub, ALICE))
    reg = load_registry(root)
    assert "not-independent" in codes(validate_registry(reg))
    assert reg.status(sub.id).state == "unverified"


def test_disagreeing_reproduction_marks_submission_disputed(root):
    sub = create_submission(MODEL, "ed", ALICE)
    write_record(root, sub)
    write_record(root, create_reproduction(sub, BOB))
    # Carol reproduces with a different (wrong) answer.
    rep = create_reproduction(sub, CAROL)
    data = rep.model_dump(mode="json")
    data["result"]["observables"]["double_occupancy"]["value"] += 0.01
    from superconductor.registry import Reproduction

    data["id"] = content_id(Reproduction.model_validate(data))
    write_record(root, Reproduction.model_validate(data))
    reg = load_registry(root)
    st = reg.status(sub.id)
    assert st.state == "disputed" and st.disagreeing == ("carol",) and st.agreeing == ("bob",)
    assert [f.code for f in validate_registry(reg) if f.level == "warning"] == ["disputed"]


def test_misplaced_and_orphan_records(root):
    sub = create_submission(MODEL, "ed", ALICE)
    path = write_record(root, sub)
    path.rename(path.with_name("renamed.json"))
    rep = create_reproduction(sub, BOB)
    write_record(root, rep)
    found = codes(validate_registry(load_registry(root)))
    assert "bad-filename" in found and "unknown-submission" not in found
    (root / "submissions" / "renamed.json").unlink()
    assert "unknown-submission" in codes(validate_registry(load_registry(root)))


def test_invalid_handle_rejected():
    with pytest.raises(ValidationError):
        Contributor(handle="not a handle!")


def test_cli_submit_reproduce_validate(root, tmp_path, capsys):
    model_path = tmp_path / "m.json"
    model_path.write_text(MODEL.to_json())
    assert main(["submit", str(model_path), "--handle", "alice", "--results", str(root)]) == 0
    sid = Path(capsys.readouterr().out.strip()).stem
    assert main(["reproduce", sid, "--handle", "alice", "--results", str(root)]) == 1
    assert main(["reproduce", sid, "--handle", "bob", "--results", str(root)]) == 0
    assert "reproduced" in capsys.readouterr().out
    assert main(["validate", "--reproduce", "--results", str(root)]) == 0
    assert main(["status", "--results", str(root)]) == 0
    assert "reproduced" in capsys.readouterr().out


def test_repository_registry_is_valid():
    """The committed results/ directory must always validate (CI also runs this)."""
    root = Path(__file__).parent.parent / "results"
    errors = [f for f in validate_registry(load_registry(root)) if f.level == "error"]
    assert errors == []
