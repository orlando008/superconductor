"""Numerical settings and convergence diagnostics are recorded with every result (issue 18)."""

import json

import pytest

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.registry import (
    Contributor,
    Reproduction,
    Submission,
    content_id,
    create_reproduction,
    create_submission,
    load_registry,
    validate_registry,
    write_record,
)
from superconductor.solvers import AtomicLimit, ExactDiagonalization, FreeFermions
from superconductor.solvers.ed import _RESIDUAL_TOL

ALICE, BOB = Contributor(handle="alice"), Contributor(handle="bob")


def model(lx=3, ly=2, nu=3, nd=2, **kw):
    params = dict(t_prime=-0.3, U=6.0) | kw
    return HubbardModel(
        lattice=SquareLattice(lx=lx, ly=ly), filling=Filling(n_up=nu, n_dn=nd), **params
    )


@pytest.mark.parametrize(
    "solver,m",
    [
        (ExactDiagonalization(), model()),
        (ExactDiagonalization(dense_limit=0), model()),
        (FreeFermions(), model(U=0.0)),
        (AtomicLimit(), model(t=0.0, t_prime=0.0)),
    ],
)
def test_every_result_records_its_solver_settings(solver, m):
    result = solver.solve(m)
    assert result.metadata["solver_settings"] == solver.settings()
    json.dumps(result.metadata["solver_settings"])  # must be serializable into records


def test_ed_settings_reflect_configuration():
    assert ExactDiagonalization(dense_limit=0).settings()["dense_limit"] == 0
    assert ExactDiagonalization().settings()["dense_limit"] == 4096
    keys = set(ExactDiagonalization().settings())
    assert {
        "dense_limit",
        "degeneracy_rtol",
        "residual_tol",
        "lanczos_tol",
        "max_degeneracy",
    } <= keys


@pytest.mark.parametrize(
    "m,dense_limit,method",
    [
        (model(), 4096, "dense"),
        (model(), 0, "lanczos+deflation"),
        (model(lx=2, ly=2, nu=2, nd=2, U=0.0, t_prime=0.0), 0, "lanczos+deflation"),  # 4-fold
        (model(t=0.0, t_prime=0.0), 4096, "diagonal"),
    ],
)
def test_ed_records_convergence_diagnostics(m, dense_limit, method):
    meta = ExactDiagonalization(dense_limit=dense_limit).solve(m).metadata
    assert meta["method"] == method
    # Every reported ground vector satisfies H v = E0 v to within the solver's tolerance.
    assert meta["max_residual"] <= _RESIDUAL_TOL
    if method == "lanczos+deflation":
        # One pass per degenerate ground state, plus the pass that finds the gap above them.
        assert meta["lanczos_passes"] == meta["ground_state_degeneracy"] + 1
    else:
        assert "lanczos_passes" not in meta


def test_submission_provenance_carries_settings_and_hardware():
    sub = create_submission(model(), "ed", ALICE)
    assert sub.provenance.solver_settings == ExactDiagonalization().settings()
    assert sub.provenance.solver_settings == sub.result.metadata["solver_settings"]
    env = sub.provenance.environment
    assert {"python", "numpy", "scipy", "machine", "cpu", "cpu_count", "blas"} <= set(env)


def test_tampering_with_recorded_settings_breaks_the_id(tmp_path):
    root = tmp_path / "results"
    path = write_record(root, create_submission(model(), "ed", ALICE))
    data = json.loads(path.read_text())
    data["provenance"]["solver_settings"]["residual_tol"] = 1.0
    path.write_text(json.dumps(data))
    codes = [f.code for f in validate_registry(load_registry(root)) if f.level == "error"]
    assert "id-mismatch" in codes


def _with_settings(record_cls, record, **changes):
    data = record.model_dump(mode="json")
    data["provenance"]["solver_settings"].update(changes)
    data["id"] = "pending"
    data["id"] = content_id(record_cls.model_validate(data))
    return record_cls.model_validate(data)


def test_reproduction_with_different_settings_gets_a_warning_not_an_error(tmp_path):
    root = tmp_path / "results"
    sub = create_submission(model(), "ed", ALICE)
    write_record(root, sub)
    rep = _with_settings(Reproduction, create_reproduction(sub, BOB), dense_limit=0)
    write_record(root, rep)
    findings = validate_registry(load_registry(root))
    assert [f.code for f in findings if f.level == "error"] == []
    warnings = [f for f in findings if f.code == "different-settings"]
    assert len(warnings) == 1 and "dense_limit" in warnings[0].message
    assert load_registry(root).status(sub.id).state == "reproduced"


def test_matching_settings_and_other_solvers_give_no_warning(tmp_path):
    root = tmp_path / "results"
    sub = create_submission(model(U=0.0), "ed", ALICE)
    write_record(root, sub)
    write_record(root, create_reproduction(sub, BOB))  # same solver, same settings
    write_record(root, create_reproduction(sub, Contributor(handle="carol"), solver_name="free"))
    assert validate_registry(load_registry(root)) == []


def test_records_without_settings_still_validate(tmp_path):
    """Records made before settings were recorded stay valid, with no warnings."""
    root = tmp_path / "results"
    old_sub = _strip(Submission, create_submission(model(), "ed", ALICE))
    write_record(root, old_sub)
    write_record(root, create_reproduction(old_sub, BOB))
    assert validate_registry(load_registry(root), reproduce=True) == []


def _strip(record_cls, record):
    data = record.model_dump(mode="json")
    data["provenance"]["solver_settings"] = {}
    data["result"]["metadata"].pop("solver_settings", None)
    for key in ("machine", "cpu", "cpu_count", "blas"):
        data["provenance"]["environment"].pop(key, None)
    data["id"] = "pending"
    data["id"] = content_id(record_cls.model_validate(data))
    return record_cls.model_validate(data)
