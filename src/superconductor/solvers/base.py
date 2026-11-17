"""The common solver interface and result format.

Every solver reads a :class:`~superconductor.model.HubbardModel` and returns a :class:`Result`
that records the model fingerprint, the solver name and version, and a set of named
observables with error bars. Observables use a shared vocabulary so results from different
methods can be compared automatically:

- ``energy``: total energy <H> (including the -mu N term)
- ``energy_per_site``: ``energy`` / N
- ``kinetic_energy``: <H_kin>
- ``double_occupancy``: (1/N) sum_i <n_{i,up} n_{i,dn}>

At T = 0 with a degenerate ground state, observables are averaged uniformly over the ground
manifold (the T -> 0+ limit), so they are well defined regardless of which basis a solver
happens to pick inside that manifold.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from superconductor.model import HubbardModel


class NotApplicableError(ValueError):
    """Raised when a solver is asked to solve a model outside its regime of validity."""


@dataclass(frozen=True)
class Observable:
    value: float
    error: float = 0.0  # one-sigma statistical/systematic estimate; 0 for exact methods


@dataclass(frozen=True)
class Result:
    solver: str
    solver_version: str
    model_fingerprint: str
    observables: dict[str, Observable]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> float:
        return self.observables[name].value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=float)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Result:
        return cls(
            solver=data["solver"],
            solver_version=data["solver_version"],
            model_fingerprint=data["model_fingerprint"],
            observables={k: Observable(**v) for k, v in data["observables"].items()},
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CostEstimate:
    """What a calculation will cost, predicted before running it.

    ``seconds`` is single-core wall time; ``memory_bytes`` is peak memory. Either is ``None``
    when the solver has no cost model. ``feasible`` is ``False`` when the solver can't run
    the model at all (``reason`` says why; numbers may still be given, to show how far out of
    reach it is), ``True`` when it can, and ``None`` when unknown. ``driver`` names what the
    cost is dominated by, in words.
    """

    solver: str
    feasible: bool | None
    seconds: float | None = None
    memory_bytes: float | None = None
    driver: str = ""
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Solver(ABC):
    """Base class for all solvers. Subclasses set ``name``/``version`` and implement two hooks.

    Optionally they also implement ``estimate`` so users and agents can see what a
    calculation will cost before starting it.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"

    def applicability(self, model: HubbardModel) -> str | None:
        """Return ``None`` if this solver can handle ``model``, otherwise a reason why not."""
        return None

    def estimate(self, model: HubbardModel) -> CostEstimate:
        """Predict time and memory for ``model``. The default knows nothing about cost."""
        reason = self.applicability(model)
        return CostEstimate(
            solver=self.name,
            feasible=False if reason else None,
            reason=reason,
            driver="no cost model for this solver yet",
        )

    @abstractmethod
    def _solve(self, model: HubbardModel) -> tuple[dict[str, Observable], dict[str, Any]]:
        """Compute observables and metadata for an applicable model."""

    def solve(self, model: HubbardModel) -> Result:
        reason = self.applicability(model)
        if reason is not None:
            raise NotApplicableError(f"{self.name}: {reason}")
        observables, metadata = self._solve(model)
        return Result(
            solver=self.name,
            solver_version=self.version,
            model_fingerprint=model.fingerprint(),
            observables=observables,
            metadata=metadata,
        )


def standard_observables(
    model: HubbardModel, energy: float, double_occupancy_total: float
) -> dict[str, Observable]:
    """Build the shared observable set from the total energy and total double occupancy."""
    n = model.num_sites
    kinetic = energy - model.U * double_occupancy_total + model.mu * model.filling.total
    return {
        "energy": Observable(energy),
        "energy_per_site": Observable(energy / n),
        "kinetic_energy": Observable(kinetic),
        "double_occupancy": Observable(double_occupancy_total / n),
    }
