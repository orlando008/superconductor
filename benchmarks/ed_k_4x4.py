"""Production runs of momentum-resolved ED (``ed-k``) on the 4x4 Hubbard cluster.

    OPENBLAS_NUM_THREADS=1 python benchmarks/ed_k_4x4.py 5 5 --U 4 --workers 4

Prints the cost estimate, then the result: energy, energy per site, double occupancy,
ground-state momenta and degeneracy, wall time and peak memory of the largest process.
Set OPENBLAS_NUM_THREADS=1 when using workers. Otherwise each worker's BLAS threads compete
for the cores and it runs slower (see solvers/ed_k.py).

Measured on a 4-core, 15 GB cloud container (U = 4, OPENBLAS_NUM_THREADS=1):

    filling  t'     workers  states  wall      largest process  E / N
    (4, 4)   0      4        3.3e6   86 s      0.4 GB           -1.09593111
    (5, 5)   0      4        1.9e7   17 min    4.4 GB           -1.22380860
    (5, 5)   -0.22  2        1.9e7   26 min    8.2 GB           -1.12616014
    (7, 7)   0      2        1.3e8   about 2 h 6.7 GB           -0.98403678
    (8, 8)   0      2        1.7e8   86 min    7.3 GB           -0.85136593

(4, 4) is the current solver; the (5, 5) rows predate spin flip and the point-group
shortcut, which make them faster now. The (7, 7) time adds the run before a container
restart, resumed from its checkpoints, to the 60 min after it.

Published comparisons:
  - (5, 5): Husslein, Fettes & Morgenstern, arXiv:cond-mat/9705026, Tables 1 and 2,
    E/N = -1.223808 and -1.126160. That also shows their sign convention for t' is the same
    as ours.
  - (8, 8), half filling: Qin, Shi & Zhang, "A benchmark study of the two-dimensional Hubbard
    model with auxiliary-field quantum Monte Carlo method" (OSTI 1565498), Table IV:
    E = -13.616(6), potential energy 7.370(8), kinetic energy -20.989(4). This run gives
    -13.6218548, 7.36804 and -20.98989: all within their error bars.
  - (7, 7): Fano, Ortolani & Parola, Phys. Rev. B 42, 6877 (1990), as summarized by Cini,
    Perfetto & Stefanucci (arXiv:cond-mat/0010292): for U/t > 3 the ground state is
    three-fold degenerate, at K = (0,0), (pi,0) and (0,pi). This run finds exactly that.

With t' != 0 the matrix has twice the non-zeros: 4 workers exceeded 15 GB and were killed,
while 2 workers fit. Run `superconductor estimate --solver ed-k` first. For long runs set
SUPERCONDUCTOR_CHECKPOINT_DIR, so an interrupted run resumes where it stopped.

To record a result in the registry, write the model to a file and run
``superconductor submit model.json --solver ed-k --handle <you> --agent <agent>``.
"""

import argparse
import json
import resource
import time

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.solvers import MomentumED


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("n_up", type=int)
    ap.add_argument("n_dn", type=int)
    ap.add_argument("--U", type=float, default=4.0)
    ap.add_argument("--t-prime", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    model = HubbardModel(
        lattice=SquareLattice(lx=4, ly=4),
        U=args.U,
        t_prime=args.t_prime,
        filling=Filling(n_up=args.n_up, n_dn=args.n_dn),
    )
    solver = MomentumED(workers=args.workers)
    est = solver.estimate(model)
    print(
        f"estimate: {est.seconds:.0f} s, {est.memory_bytes / 1e9:.1f} GB ({est.driver})", flush=True
    )
    start = time.time()
    res = solver.solve(model)
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    print(
        json.dumps(
            {
                "model": model.physics_dict(),
                "energy": res["energy"],
                "energy_per_site": res["energy_per_site"],
                "double_occupancy": res["double_occupancy"],
                "ground_momenta": res.metadata["ground_momenta"],
                "degeneracy": res.metadata["ground_state_degeneracy"],
                "max_residual": res.metadata["max_residual"],
                "wall_seconds": round(time.time() - start, 1),
                "peak_memory_gb_largest_process": round(peak / 1e6, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
