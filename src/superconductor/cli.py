"""Command-line entry point: ``superconductor {solve,crosscheck,fingerprint,solvers}``."""

from __future__ import annotations

import argparse
import json
import sys

from superconductor.model import load_model
from superconductor.solvers import get_solver, list_solvers
from superconductor.validation import run_suite, sanity_check


def _solve(args) -> int:
    model = load_model(args.model)
    result = get_solver(args.solver).solve(model)
    checks = sanity_check(model, result)
    payload = result.to_dict()
    payload["sanity_checks"] = [c.__dict__ for c in checks]
    print(json.dumps(payload, indent=2, default=float))
    return 0 if all(c.passed for c in checks) else 1


def _crosscheck(args) -> int:
    reports = run_suite()
    failures = 0
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        solvers = "+".join(r.results)
        print(f"{status}  {r.model.name:<40} {solvers:<10} {len(r.comparisons)} comparisons")
        for c in r.failures:
            failures += 1
            print(
                f"      {c.observable}: {c.solver_a}={c.value_a:.12g} "
                f"{c.solver_b}={c.value_b:.12g} |diff|={c.difference:.3g} > {c.tolerance:.3g}"
            )
    total = sum(len(r.comparisons) for r in reports)
    print(f"\n{len(reports)} models, {total} comparisons, {failures} disagreements")
    return 0 if failures == 0 else 1


def _fingerprint(args) -> int:
    print(load_model(args.model).fingerprint())
    return 0


def _solvers(args) -> int:
    print("\n".join(list_solvers()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="superconductor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", help="Solve a model file and run sanity checks on the result")
    p.add_argument("model", help="Path to a model JSON file")
    p.add_argument("--solver", default="ed", choices=list_solvers())
    p.set_defaults(func=_solve)

    p = sub.add_parser("crosscheck", help="Run the standard cross-validation suite")
    p.set_defaults(func=_crosscheck)

    p = sub.add_parser("fingerprint", help="Print the content hash of a model file")
    p.add_argument("model")
    p.set_defaults(func=_fingerprint)

    p = sub.add_parser("solvers", help="List registered solvers")
    p.set_defaults(func=_solvers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
