"""Shear-stress ablation + sediment transport.

The coupling between fluid and stone. The current flows through the *air*
around the glyph; where it scrapes along the stone surface it tears material
free. Ablated mass becomes ``sediment``, which rides the current and settles
where the flow can no longer carry it — building lee-side sandbars.

Erosion is evaluated on the fluid side of the air/stone interface: for each
air voxel adjacent to stone, the local speed (and surface curvature) sets how
much mass is pulled from the neighboring stone into the sediment field. This
is what turns a blocky extrusion into a river-worn form, gradually, across
all iterations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, laplace

from .world import World


def _advect_scalar(field: np.ndarray, u, v, w, dt: float) -> np.ndarray:
    nx, ny, nz = field.shape
    X, Y, Z = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([X - u * dt, Y - v * dt, Z - w * dt]).reshape(3, -1)
    out = map_coordinates(field, coords, order=1, mode="nearest", prefilter=False)
    return out.reshape(field.shape).astype(np.float32)


def _neighbor_max(field: np.ndarray) -> np.ndarray:
    """Max over the 6 face-neighbors — used to pull stone toward the flow."""
    m = field.copy()
    m[1:] = np.maximum(m[1:], field[:-1])
    m[:-1] = np.maximum(m[:-1], field[1:])
    m[:, 1:] = np.maximum(m[:, 1:], field[:, :-1])
    m[:, :-1] = np.maximum(m[:, :-1], field[:, 1:])
    m[:, :, 1:] = np.maximum(m[:, :, 1:], field[:, :, :-1])
    m[:, :, :-1] = np.maximum(m[:, :, :-1], field[:, :, 1:])
    return m


def _air_adjacent_to_stone(stone: np.ndarray, lo: float = 0.5) -> np.ndarray:
    """Air voxels that touch stone — the cells where the flow does the scraping."""
    air = stone < lo
    adj = np.zeros_like(stone, dtype=bool)
    adj[1:] |= stone[:-1] >= lo
    adj[:-1] |= stone[1:] >= lo
    adj[:, 1:] |= stone[:, :-1] >= lo
    adj[:, :-1] |= stone[:, 1:] >= lo
    adj[:, :, 1:] |= stone[:, :, :-1] >= lo
    adj[:, :, :-1] |= stone[:, :, 1:] >= lo
    return air & adj


class ErosionModel:
    def __init__(
        self,
        world: World,
        erodability: float = 0.012,    # base rate — sustained wear
        speed_scale: float = 3.0,      # s0: speed at which erosion is "1x"
        shear_exponent: float = 1.3,   # nonlinearity of speed->erosion
        curvature_gain: float = 4.0,   # how much protrusions are preferred
        capacity: float = 0.8,         # max sediment the current can carry
        deposit_rate: float = 0.06,    # how fast sediment settles when slow
        detach_rate: float = 0.35,     # how fast settled sediment re-bonds
        dt: float = 0.25,
    ) -> None:
        self.w = world
        self.k = np.float32(erodability)
        self.s0 = np.float32(speed_scale)
        self.exp = np.float32(shear_exponent)
        self.curv = np.float32(curvature_gain)
        self.cap = np.float32(capacity)
        self.dep = np.float32(deposit_rate)
        self.det = np.float32(detach_rate)
        self.dt = np.float32(dt)

    def step(self) -> None:
        w = self.w
        dt = self.dt

        speed = np.sqrt(w.u ** 2 + w.v ** 2 + w.w ** 2).astype(np.float32)

        # Surface curvature: -laplacian is positive on convex protrusions
        # (bumps into the flow) — they erode faster than flats.
        curv = -laplace(w.stone).astype(np.float32)
        curv_bonus = self.curv * np.clip(curv, 0.0, None)

        # Erosion is evaluated on the FLUID side: air cells touching stone.
        attack = _air_adjacent_to_stone(w.stone).astype(np.float32)
        # The stone mass available to be pulled into each attacking air cell:
        stone_nearby = _neighbor_max(w.stone)

        speed_factor = (speed / self.s0) ** self.exp
        erosion = self.k * speed_factor * (1.0 + curv_bonus) * attack * stone_nearby
        erosion = np.clip(erosion, 0.0, 0.04) * dt * 8.0

        # Pull that mass out of the adjacent stone. Distribute the removal
        # back onto the stone field by dilating the erosion onto stone cells.
        # Simplest mass-conserving approach: subtract a blurred erosion from
        # stone, weighted by where stone actually is.
        from scipy.ndimage import maximum_filter
        removal = maximum_filter(erosion, size=3)
        w.stone = np.clip(w.stone - removal * (w.stone > 0).astype(np.float32),
                          0.0, 1.0).astype(np.float32)
        w.sediment += erosion  # ablated mass enters suspension

        # --- advect suspended sediment with the current ------------------
        w.sediment = _advect_scalar(w.sediment, w.u, w.v, w.w, dt)
        w.sediment[w.stone > 0.6] *= 0.15  # can't reside inside solid rock

        # --- transport capacity & deposition -----------------------------
        # Deposited sediment is removed from the system (carried out of the
        # domain) rather than re-bonding as fresh stone — the word wears away
        # to nothing and stays gone. No second body grows from the silt.
        load = self.cap * (speed / (speed + 2.5)).astype(np.float32)
        excess = np.clip(w.sediment - load, 0.0, None)
        deposit = self.dep * excess * dt
        w.sediment -= deposit
        np.clip(w.sediment, 0.0, None, out=w.sediment)
        # (deposit is discarded — it leaves the domain as suspended outflow)
