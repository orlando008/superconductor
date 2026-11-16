"""The single-band t-t'-U Hubbard model specification.

Hamiltonian (all energies in the same units as ``t``):

    H = - sum_{<ij>,s}  t  (c+_{is} c_{js} + h.c.)
        - sum_{<<ij>>,s} t' (c+_{is} c_{js} + h.c.)
        + U sum_i n_{i,up} n_{i,dn}
        - mu sum_{i,s} n_{is}

With this sign convention cuprates have t > 0 and t' < 0 (typically t'/t ~ -0.15 to -0.45).

This is schema version 0.1. Multi-orbital (three-band Emery) models, longer-range hopping
and interactions are planned; the schema version will be bumped when the format changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from superconductor.model.lattice import SquareLattice

SCHEMA_VERSION = "0.1"


class Filling(BaseModel):
    """Fixed particle numbers per spin species (canonical ensemble)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_up: int = Field(ge=0)
    n_dn: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.n_up + self.n_dn


class HubbardModel(BaseModel):
    """A fully specified Hubbard model instance: what to solve, not how to solve it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    model: Literal["hubbard"] = "hubbard"
    name: str | None = Field(default=None, description="Human label; not part of the physics.")
    lattice: SquareLattice
    t: float = 1.0
    t_prime: float = 0.0
    U: float = 0.0
    mu: float = 0.0
    filling: Filling
    temperature: float = Field(default=0.0, ge=0.0, description="0 means ground state.")

    @model_validator(mode="after")
    def _check_filling(self) -> HubbardModel:
        n = self.lattice.num_sites
        if self.filling.n_up > n or self.filling.n_dn > n:
            raise ValueError(
                f"Filling ({self.filling.n_up} up, {self.filling.n_dn} dn) exceeds {n} sites"
            )
        return self

    # ----- derived quantities -------------------------------------------------------------

    @property
    def num_sites(self) -> int:
        return self.lattice.num_sites

    @property
    def density(self) -> float:
        """Electrons per site, n."""
        return self.filling.total / self.num_sites

    @property
    def hole_doping(self) -> float:
        """Hole doping p = 1 - n (negative means electron doping)."""
        return 1.0 - self.density

    def hopping_matrix(self) -> np.ndarray:
        """Single-particle matrix h with H_kin = sum_{ij,s} h_ij c+_{is} c_{js}."""
        n = self.num_sites
        h = np.zeros((n, n))
        for shell, amplitude in ((1, self.t), (2, self.t_prime)):
            if amplitude == 0.0:
                continue
            for i, j in self.lattice.bonds(shell):
                h[i, j] -= amplitude
                h[j, i] -= amplitude
        return h

    # ----- provenance ----------------------------------------------------------------------

    def physics_dict(self) -> dict:
        """Canonical content defining the physics (excludes the human-readable name)."""
        return self.model_dump(mode="json", exclude={"name"})

    def fingerprint(self) -> str:
        """Stable content hash; two specs with the same fingerprint describe the same problem."""
        canonical = json.dumps(self.physics_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(indent=2, **kwargs)

    def with_updates(self, **changes) -> HubbardModel:
        """Return a validated copy with some fields changed."""
        return HubbardModel.model_validate({**self.model_dump(), **changes})


def load_model(path: str | Path) -> HubbardModel:
    return HubbardModel.model_validate_json(Path(path).read_text())
