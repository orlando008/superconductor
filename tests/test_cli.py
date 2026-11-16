import json
from pathlib import Path

from superconductor.cli import main

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_solve_example(capsys):
    assert main(["solve", str(EXAMPLES / "hubbard_2x2_half_filling.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["solver"] == "ed"
    assert out["model_fingerprint"].startswith("sha256:")
    assert all(c["passed"] for c in out["sanity_checks"])


def test_fingerprint_and_solvers(capsys):
    assert main(["fingerprint", str(EXAMPLES / "hubbard_4x2_hole_doped.json")]) == 0
    assert capsys.readouterr().out.startswith("sha256:")
    assert main(["solvers"]) == 0
    assert set(capsys.readouterr().out.split()) == {"atomic", "ed", "free"}
