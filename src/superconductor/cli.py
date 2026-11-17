"""Command-line entry point for the workbench.

Commands: solve, estimate, crosscheck, fingerprint, solvers, submit, reproduce, validate,
status, check-authorship.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from superconductor.model import load_model
from superconductor.solvers import CostEstimate, get_solver, list_solvers
from superconductor.validation import run_suite, sanity_check

DEFAULT_BUDGET_SECONDS = 600.0
DEFAULT_BUDGET_GB = 8.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "unknown"
    if s < 1:
        return "under a second"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.0f} days"


def _fmt_bytes(b: float | None) -> str:
    if b is None:
        return "unknown"
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= size:
            return f"{b / size:.1f} {unit}"
    return f"{b:.0f} B"


def budget_warnings(est: CostEstimate, seconds: float, gigabytes: float) -> list[str]:
    """Human-readable warnings for an estimate that exceeds the time or memory budget."""
    out = []
    if est.seconds is not None and est.seconds > seconds:
        out.append(
            f"estimated time {_fmt_seconds(est.seconds)} exceeds the budget of "
            f"{_fmt_seconds(seconds)} ({est.driver})"
        )
    if est.memory_bytes is not None and est.memory_bytes > gigabytes * 1e9:
        out.append(
            f"estimated memory {_fmt_bytes(est.memory_bytes)} exceeds the budget of "
            f"{_fmt_bytes(gigabytes * 1e9)} ({est.driver})"
        )
    return out


def _warn_if_over_budget(solver, model, args) -> None:
    for w in budget_warnings(solver.estimate(model), args.budget_seconds, args.budget_gb):
        print(f"Warning: {w}", file=sys.stderr)


def _estimate(args) -> int:
    model = load_model(args.model)
    est = get_solver(args.solver).estimate(model)
    warnings = budget_warnings(est, args.budget_seconds, args.budget_gb)
    if args.json:
        payload = est.to_dict()
        payload["over_budget"] = warnings
        print(json.dumps(payload, indent=2))
        return 0
    feasible = {True: "yes", False: "no", None: "unknown"}[est.feasible]
    print(f"Solver:     {est.solver}")
    print(f"Can run:    {feasible}" + (f" ({est.reason})" if est.reason else ""))
    print(f"Driven by:  {est.driver}")
    print(f"Time:       {_fmt_seconds(est.seconds)} (one CPU core)")
    print(f"Memory:     {_fmt_bytes(est.memory_bytes)}")
    for w in warnings:
        print(f"Warning:    {w}")
    return 0


def _solve(args) -> int:
    model = load_model(args.model)
    solver = get_solver(args.solver)
    _warn_if_over_budget(solver, model, args)
    result = solver.solve(model)
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

    from superconductor.validation import run_hamiltonian_checks

    checks = run_hamiltonian_checks()
    mismatches = [c for c in checks if not c.passed]
    for c in mismatches:
        print(f"FAIL  Hamiltonian {c.model_name}: {c.detail}")
    print(
        f"{len(checks)} Hamiltonians checked against the independent reference, "
        f"{len(mismatches)} mismatches"
    )

    from superconductor.validation import run_perturbative_checks

    outcomes = run_perturbative_checks()
    pert_failures = [o for o in outcomes if o.status == "fail"]
    for o in pert_failures:
        print(f"FAIL  {o.name} {o.model_name}: {o.detail}")
    passed = sum(o.status == "pass" for o in outcomes)
    skipped = sum(o.status == "skipped" for o in outcomes)
    print(
        f"{len(outcomes)} perturbative-limit checks: {passed} passed, {skipped} not applicable, "
        f"{len(pert_failures)} failed"
    )
    return 0 if failures == 0 and not mismatches and not pert_failures else 1


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
    _warn_if_over_budget(get_solver(args.solver), model, args)
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
    if sub.contributor.handle.lower() == args.handle.lower():
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


def _check_authorship(args) -> int:
    from superconductor.registry import check_added_records_authorship

    if args.added_files == "-":
        lines = sys.stdin.read().splitlines()
    else:
        with open(args.added_files) as fh:
            lines = fh.read().splitlines()
    findings = check_added_records_authorship(lines, args.author, repo_root=args.repo_root)
    for f in findings:
        print(f)
    n_records = sum(1 for line in lines if line.strip().endswith(".json"))
    print(
        f"{n_records} added record file(s) checked against author {args.author!r}: "
        f"{len(findings)} problem(s)"
    )
    return 1 if findings else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="superconductor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_budget_args(p):
        p.add_argument(
            "--budget-seconds",
            type=float,
            default=_env_float("SUPERCONDUCTOR_BUDGET_SECONDS", DEFAULT_BUDGET_SECONDS),
            help="Warn if the estimated time exceeds this (env SUPERCONDUCTOR_BUDGET_SECONDS)",
        )
        p.add_argument(
            "--budget-gb",
            type=float,
            default=_env_float("SUPERCONDUCTOR_BUDGET_GB", DEFAULT_BUDGET_GB),
            help="Warn if the estimated memory exceeds this (env SUPERCONDUCTOR_BUDGET_GB)",
        )

    p = sub.add_parser("solve", help="Solve a model file and run sanity checks on the result")
    p.add_argument("model", help="Path to a model JSON file")
    p.add_argument("--solver", default="ed", choices=list_solvers())
    add_budget_args(p)
    p.set_defaults(func=_solve)

    p = sub.add_parser("estimate", help="Predict time and memory before solving")
    p.add_argument("model", help="Path to a model JSON file")
    p.add_argument("--solver", default="ed", choices=list_solvers())
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    add_budget_args(p)
    p.set_defaults(func=_estimate)

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
    add_budget_args(p)
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

    p = sub.add_parser(
        "check-authorship",
        help="Check that added result records name the given PR author (used by CI)",
    )
    p.add_argument("--author", required=True, help="GitHub login of the pull request author")
    p.add_argument(
        "--added-files", default="-", help="File listing added paths, one per line ('-' = stdin)"
    )
    p.add_argument("--repo-root", default=".", help="Root the listed paths are relative to")
    p.set_defaults(func=_check_authorship)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
