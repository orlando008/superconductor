"""Command-line entry point for the workbench.

Commands: solve, crosscheck, fingerprint, solvers, submit, reproduce, validate, status.
"""

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


def _contributor(args):
    from superconductor.registry import Contributor

    return Contributor(handle=args.handle, agent=args.agent)


def _submit(args) -> int:
    from superconductor.registry import create_submission, write_record

    model = load_model(args.model)
    sub = create_submission(model, args.solver, _contributor(args), notes=args.notes)
    checks = sanity_check(model, sub.result.to_result())
    if not all(c.passed for c in checks):
        print("Refusing to write a submission that fails sanity checks:", file=sys.stderr)
        for c in checks:
            print(f"  {c.name}: {c.detail}", file=sys.stderr)
        return 1
    print(write_record(args.results, sub))
    return 0


def _reproduce(args) -> int:
    from superconductor.registry import create_reproduction, load_registry, write_record

    reg = load_registry(args.results)
    if args.submission_id not in reg.submissions:
        print(f"No submission {args.submission_id} in {args.results}", file=sys.stderr)
        return 1
    _, sub = reg.submissions[args.submission_id]
    if sub.contributor.handle == args.handle:
        print("A reproduction must come from a different contributor.", file=sys.stderr)
        return 1
    rep = create_reproduction(sub, _contributor(args), solver_name=args.solver, notes=args.notes)
    path = write_record(args.results, rep)
    reg = load_registry(args.results)
    print(path)
    print(f"submission {sub.id} is now: {reg.status(sub.id).state}")
    return 0


def _validate(args) -> int:
    from superconductor.registry import load_registry, validate_registry

    reg = load_registry(args.results)
    findings = validate_registry(reg, reproduce=args.reproduce)
    for f in findings:
        print(f)
    errors = sum(f.level == "error" for f in findings)
    n_rep = sum(len(v) for v in reg.reproductions.values())
    print(
        f"{len(reg.submissions)} submissions, {n_rep} reproductions: "
        f"{errors} errors, {len(findings) - errors} warnings"
    )
    return 1 if errors else 0


def _status(args) -> int:
    from superconductor.registry import load_registry

    reg = load_registry(args.results)
    for sid, (_, sub) in sorted(reg.submissions.items()):
        st = reg.status(sid)
        label = sub.model.name or sub.model.fingerprint()[:19]
        print(
            f"{sid}  {st.state:<10} agree={len(st.agreeing)} "
            f"(other method: {len(st.agreeing_other_method)}) "
            f"disagree={len(st.disagreeing)}  {sub.provenance.solver:<8} {label}"
        )
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

    def add_registry_args(p, contributor=False):
        p.add_argument("--results", default="results", help="Registry root (default: results)")
        if contributor:
            p.add_argument("--handle", required=True, help="GitHub username of responsible human")
            p.add_argument("--agent", default=None, help="AI agent/model used, if any")
            p.add_argument("--notes", default=None)

    p = sub.add_parser("submit", help="Solve a model and record the result in the registry")
    p.add_argument("model")
    p.add_argument("--solver", default="ed", choices=list_solvers())
    add_registry_args(p, contributor=True)
    p.set_defaults(func=_submit)

    p = sub.add_parser("reproduce", help="Independently re-run a submission and record it")
    p.add_argument("submission_id")
    p.add_argument(
        "--solver",
        default=None,
        choices=list_solvers(),
        help="Defaults to the submission's solver; another method is stronger",
    )
    add_registry_args(p, contributor=True)
    p.set_defaults(func=_reproduce)

    p = sub.add_parser("validate", help="Validate every record in the results registry")
    p.add_argument("--reproduce", action="store_true", help="Also re-run cheap submissions")
    add_registry_args(p)
    p.set_defaults(func=_validate)

    p = sub.add_parser("status", help="Show the verification status of every submission")
    add_registry_args(p)
    p.set_defaults(func=_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
