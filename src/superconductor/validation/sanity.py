"""Automatic sanity checks on any result (seed of layer 7).

Each check is a rigorous inequality or identity, so a failure always means something is
wrong with the result, not merely that it is surprising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from superconductor.model import HubbardModel
from superconductor.validation.reference import reference_free_ground_state

if TYPE_CHECKING:  # type only: the checks never run solver code
    from superconductor.solvers import Result


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    detail: str


def _bound(name, low, value, high, slack) -> CheckOutcome:
    ok = low - slack <= value <= high + slack
    return CheckOutcome(name, ok, f"{low:.10g} <= {value:.10g} <= {high:.10g}")


def sanity_check(model: HubbardModel, result: Result, slack: float = 1e-8) -> list[CheckOutcome]:
    if result.model_fingerprint != model.fingerprint():
        return [CheckOutcome("fingerprint", False, "result was computed for a different model")]

    out: list[CheckOutcome] = []
    obs = result.observables
    n = model.num_sites
    nu, nd = model.filling.n_up, model.filling.n_dn

    if "double_occupancy" in obs:
        low = max(0, nu + nd - n) / n
        high = min(nu, nd) / n
        out.append(
            _bound(
                "double_occupancy_range",
                low,
                obs["double_occupancy"].value,
                high,
                slack + 2 * obs["double_occupancy"].error,
            )
        )

    if "energy" in obs and "energy_per_site" in obs:
        e, e_ps = obs["energy"].value, obs["energy_per_site"].value
        out.append(
            CheckOutcome(
                "energy_per_site_consistent",
                abs(e / n - e_ps) <= slack,
                f"E/N={e / n:.10g}, energy_per_site={e_ps:.10g}",
            )
        )

    # Ground-state energy bounds for repulsive U, from the U = 0 problem:
    #   E0(0) <= E0(U)                     (U * D is a non-negative operator)
    #   E0(U) <= E_free + U * D_free       (variational, using the free ground manifold)
    # The free-fermion reference is computed independently of the solvers (see reference.py).
    if model.temperature == 0 and model.U >= 0 and "energy" in obs:
        e_free, d_free = reference_free_ground_state(model)
        upper = e_free + model.U * d_free
        out.append(
            _bound(
                "energy_free_fermion_bounds",
                e_free,
                obs["energy"].value,
                upper,
                slack + 2 * obs["energy"].error,
            )
        )
    return out
