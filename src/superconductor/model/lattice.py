"""Lattice geometry.

Sites of an ``lx`` x ``ly`` square lattice are indexed ``i = x + lx * y``.

Bond convention for periodic boundaries: each site contributes its +x, +y (shell 1) and
+x+y, +x-y (shell 2) bonds, and contributions to the same pair of sites *accumulate*. On a
periodic length-2 direction the +x and -x neighbours coincide, so the bond appears twice.
This is deliberate: it makes the real-space hopping matrix reproduce the tight-binding
dispersion ``-2t(cos kx + cos ky) - 4t' cos kx cos ky`` exactly on the allowed momenta, for any
lattice size. A direction of length 1 has no bonds along it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Boundary = Literal["periodic", "open"]

_SHELL_DISPLACEMENTS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((1, 0), (0, 1)),  # nearest neighbours (t)
    2: ((1, 1), (1, -1)),  # diagonal next-nearest neighbours (t')
}


class SquareLattice(BaseModel):
    """A finite square lattice (the CuO2 plane at the single-band level)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["square"] = "square"
    lx: int = Field(ge=1, description="Number of sites along x.")
    ly: int = Field(ge=1, description="Number of sites along y.")
    boundary_x: Boundary = "periodic"
    boundary_y: Boundary = "periodic"

    @property
    def num_sites(self) -> int:
        return self.lx * self.ly

    def site_index(self, x: int, y: int) -> int:
        return x + self.lx * y

    def coordinates(self, i: int) -> tuple[int, int]:
        return i % self.lx, i // self.lx

    def _wrap(self, coord: int, length: int, boundary: Boundary) -> int | None:
        if 0 <= coord < length:
            return coord
        if boundary == "periodic":
            return coord % length
        return None

    def bonds(self, shell: int) -> Iterator[tuple[int, int]]:
        """Yield directed-once site pairs ``(i, j)`` for the given neighbour shell.

        A pair may be yielded more than once on small periodic lattices; see module docstring.
        """
        if shell not in _SHELL_DISPLACEMENTS:
            raise ValueError(f"Unsupported neighbour shell {shell}")
        for dx, dy in _SHELL_DISPLACEMENTS[shell]:
            if (dx != 0 and self.lx == 1) or (dy != 0 and self.ly == 1):
                continue
            for y in range(self.ly):
                for x in range(self.lx):
                    nx = self._wrap(x + dx, self.lx, self.boundary_x)
                    ny = self._wrap(y + dy, self.ly, self.boundary_y)
                    if nx is None or ny is None:
                        continue
                    i, j = self.site_index(x, y), self.site_index(nx, ny)
                    if i != j:
                        yield i, j

    def allowed_momenta(self) -> np.ndarray:
        """Momenta ``(kx, ky)`` compatible with fully periodic boundaries."""
        if self.boundary_x != "periodic" or self.boundary_y != "periodic":
            raise ValueError("Momentum is only a good quantum number with periodic boundaries")
        kx = 2 * np.pi * np.arange(self.lx) / self.lx
        ky = 2 * np.pi * np.arange(self.ly) / self.ly
        return np.array([(a, b) for b in ky for a in kx])
