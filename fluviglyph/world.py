"""The virtual riverbed: a voxel grid holding stone, sediment, and a current."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .word import rasterize_word


@dataclass
class World:
    """A 3D voxel domain.

    Axes:
        x : flow direction  (the river runs +x)
        y : vertical
        z : depth (extrusion direction of the glyph)

    Fields (all float32, shape ``(nx, ny, nz)``):
        stone    : solid mass in [0,1]. 1 = fresh blocky glyph, 0 = air.
        sediment : mobile ablated material suspended in the current.
        u, v, w  : velocity components (voxels/sec).
        p        : pressure (used by the projection step).
    """

    nx: int
    ny: int
    nz: int
    stone: np.ndarray = field(init=False)
    sediment: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.nx, self.ny, self.nz)
        self.stone = np.zeros(shape, np.float32)
        self.sediment = np.zeros(shape, np.float32)
        self.u = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.w = np.zeros(shape, np.float32)
        self.p = np.zeros(shape, np.float32)

    @classmethod
    def from_word(
        cls,
        word: str,
        nx: int = 128,
        ny: int = 64,
        nz: int = 48,
        inflow: float = 6.0,
    ) -> "World":
        """Build a world whose stone is an extruded blocky ``word``.

        The glyph occupies a slice of the domain in x, extruded fully in z, and
        centered in y. A steady inflow is seeded along the -x face.
        """
        world = cls(nx, ny, nz)

        # Rasterize the word into the x-y plane at the full extrusion depth.
        # rasterize_word returns (height, width) == (ny, nx); transpose to (nx, ny).
        glyph = rasterize_word(word, width=nx, height=ny).astype(np.float32)
        glyph = np.ascontiguousarray(glyph.T)

        # Place it solid through z, with a slight front-to-back taper so the
        # river meets a leading edge rather than a flat wall.
        z_profile = np.linspace(1.0, 0.92, nz, dtype=np.float32)
        world.stone[:] = glyph[:, :, None] * z_profile[None, None, :]

        # A gentle steady current filling the domain. The stone will deflect it.
        world.u[:] = inflow
        return world

    # -- geometry helpers -------------------------------------------------
    @property
    def fluid_mask(self) -> np.ndarray:
        """Cells that are open to the current (stone < 0.5)."""
        return self.stone < 0.5

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)
