"""Production runs of momentum-resolved ED (``ed-k``) on the 4x4 Hubbard cluster.

    OPENBLAS_NUM_THREADS=1 python benchmarks/ed_k_4x4.py 5 5 --U 4 --workers 4

Prints the cost estimate, then the result: energy, energy per site, double occupancy,
ground-state momenta and degeneracy, wall time and peak memory of the largest process.
Set OPENBLAS_NUM_THREADS=1 when using workers. Otherwise each worker's BLAS threads compete
for the cores and it runs slower (see solvers/ed_k.py).

Measured on a 4-core, 15 GB cloud container (U = 4, t' = 0, workers = 4):

    (4, 4)   3.3e6 states,  16 blocks of ~2.1e5:  98 s,  0.9 GB largest process
    (5, 5)   1.9e7 states,  16 blocks of ~1.2e6:  see the pull request for issue 6

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
