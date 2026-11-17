"""Measure solver cost against the predictions of ``Solver.estimate``.

    python benchmarks/ed_cost.py            # table of predicted vs measured time and memory
    python benchmarks/ed_cost.py --check    # exit 1 if any prediction is off by more than 3x
    python benchmarks/ed_cost.py --fit      # suggest ED cost constants for this machine
    python benchmarks/ed_cost.py --quick    # skip the slowest case (about 35 s)

Runs on a single thread, like the estimates. Wall time depends on the hardware, so this is a
benchmark you run deliberately, not part of the test suite. Memory is measured as the peak
of Python-visible allocations (tracemalloc), which includes NumPy and SciPy arrays.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # must be set before NumPy loads

import argparse  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402

from superconductor import Filling, HubbardModel, SquareLattice  # noqa: E402
from superconductor.solvers import get_solver  # noqa: E402
from superconductor.solvers.ed import COST, sector_dimension  # noqa: E402

FACTOR = 3.0


def model(lx, ly, nu, nd, *, t=1.0, tp=-0.3, U=6.0, T=0.0):
    return HubbardModel(
        lattice=SquareLattice(lx=lx, ly=ly),
        t=t,
        t_prime=tp if t else 0.0,
        U=U,
        temperature=T,
        filling=Filling(n_up=nu, n_dn=nd),
    )


# (label, solver, model, slow) spanning every path the estimates model.
CASES = [
    ("ed dense, small", "ed", model(3, 2, 3, 3), False),
    ("ed dense", "ed", model(4, 2, 3, 2), False),
    ("ed dense, near limit", "ed", model(4, 2, 3, 3), False),
    ("ed finite T", "ed", model(4, 2, 3, 3, T=1.0), False),
    ("ed finite T, larger", "ed", model(4, 2, 4, 4, T=1.0), False),
    ("ed diagonal (t = 0)", "ed", model(4, 2, 4, 4, t=0.0), False),
    ("ed Lanczos, small", "ed", model(4, 2, 4, 4), False),
    ("ed Lanczos", "ed", model(3, 3, 4, 4), False),
    ("ed Lanczos, 12 sites", "ed", model(4, 3, 4, 4), True),
    ("free, ground state", "free", model(10, 10, 45, 45, U=0.0), False),
    ("free, finite T", "free", model(10, 10, 45, 45, U=0.0, T=0.5), False),
    ("atomic", "atomic", model(10, 10, 55, 55, t=0.0, T=0.5), False),
]


def measure(solver_name, m):
    solver = get_solver(solver_name)
    tracemalloc.start()
    start = time.perf_counter()
    solver.solve(m)
    seconds = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return seconds, peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help=f"fail if off by more than {FACTOR}x")
    ap.add_argument("--fit", action="store_true", help="suggest ED cost constants")
    ap.add_argument("--quick", action="store_true", help="skip the slowest case")
    args = ap.parse_args(argv)

    rows, failures = [], []
    print(
        f"{'case':<24} {'pred s':>8} {'meas s':>8} {'ratio':>6}   "
        f"{'pred MB':>8} {'meas MB':>8} {'ratio':>6}"
    )
    for label, solver_name, m, slow in CASES:
        if slow and args.quick:
            continue
        est = get_solver(solver_name).estimate(m)
        seconds, peak = measure(solver_name, m)
        t_ratio = est.seconds / seconds
        m_ratio = est.memory_bytes / max(peak, 1)
        rows.append((label, solver_name, m, est, seconds, peak))
        print(
            f"{label:<24} {est.seconds:8.3f} {seconds:8.3f} {t_ratio:6.2f}   "
            f"{est.memory_bytes / 1e6:8.1f} {peak / 1e6:8.1f} {m_ratio:6.2f}"
        )
        # Below 0.1 s or 1 MB, fixed overheads dominate and ratios mean little.
        if seconds > 0.1 and not (1 / FACTOR <= t_ratio <= FACTOR):
            failures.append(f"{label}: time ratio {t_ratio:.2f}")
        if peak > 1e6 and not (1 / FACTOR <= m_ratio <= FACTOR):
            failures.append(f"{label}: memory ratio {m_ratio:.2f}")

    if args.fit:
        dense = [
            s / sector_dimension(m) ** 3
            for lbl, n, m, e, s, p in rows
            if n == "ed" and "dense" in e.details["method"] and s > 0.5
        ]
        lanczos = [
            s / e.details["estimated_nonzeros"]
            for lbl, n, m, e, s, p in rows
            if n == "ed" and e.details["method"].startswith("Lanczos") and s > 0.5
        ]
        print("\nSuggested constants for this machine (current values in brackets):")
        if dense:
            print(
                f"  dense_seconds_per_dim3   = {statistics.median(dense):.3g}"
                f"  [{COST['dense_seconds_per_dim3']:.3g}]"
            )
        if lanczos:
            print(
                f"  lanczos_seconds_per_nnz  = {statistics.median(lanczos):.3g}"
                f"  [{COST['lanczos_seconds_per_nnz']:.3g}]"
            )

    if args.check:
        if failures:
            print(f"\nFAIL: predictions off by more than {FACTOR}x:\n  " + "\n  ".join(failures))
            return 1
        print(f"\nPASS: every prediction within {FACTOR}x of the measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
