"""A 3D stable-fluids Navier-Stokes solver on a regular grid.

Implements the classic Stam (1999) "Stable Fluids" operator splitting:

    1. advect velocity (semi-Lagrangian, unconditionally stable)
    2. diffuse velocity (implicit, Jacobi) — kept light; the erosion model
       provides most of the "viscosity" via sediment drag
    3. project to a divergence-free field (pressure Poisson, Jacobi)

The stone is treated as a soft body-force drag: cells with high stone density
slow the fluid, which is where the coupling to erosion lives.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .world import World


def _advect(field: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
            dt: float, mode: str = "nearest") -> np.ndarray:
    """Semi-Lagrangian advection: new[i] = old[i - vel*dt]."""
    nx, ny, nz = field.shape
    X, Y, Z = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    bx = X - u * dt
    by = Y - v * dt
    bz = Z - w * dt
    coords = np.stack([bx, by, bz]).reshape(3, -1)
    out = map_coordinates(field, coords, order=1, mode=mode, prefilter=False)
    return out.reshape(field.shape).astype(np.float32)


def _laplacian(f: np.ndarray) -> np.ndarray:
    """7-point discrete Laplacian (interior only; boundaries zeroed)."""
    lap = -6.0 * f
    lap[1:] += f[:-1]
    lap[:-1] += f[1:]
    lap[:, 1:] += f[:, :-1]
    lap[:, :-1] += f[:, 1:]
    lap[:, :, 1:] += f[:, :, :-1]
    lap[:, :, :-1] += f[:, :, 1:]
    return lap


class FluidSolver:
    def __init__(self, world: World, viscosity: float = 1e-5,
                 jacobi_proj: int = 8, jacobi_diff: int = 4) -> None:
        self.w = world
        self.nu = np.float32(viscosity)
        self.jacobi_proj = jacobi_proj
        self.jacobi_diff = jacobi_diff

    # -- boundary conditions ---------------------------------------------
    def _apply_bounds(self) -> None:
        w = self.w
        # Inflow on -x face: a steady river.
        w.u[0, :, :] = 6.0
        w.v[0, :, :] = 0.0
        w.w[0, :, :] = 0.0
        # Outflow on +x face: copy the previous cell (free exit).
        w.u[-1, :, :] = w.u[-2, :, :]
        w.v[-1, :, :] = w.v[-2, :, :]
        w.w[-1, :, :] = w.w[-2, :, :]
        # Walls in y and z: free-slip (zero normal velocity).
        w.v[:, 0, :] = 0.0
        w.v[:, -1, :] = 0.0
        w.w[:, :, 0] = 0.0
        w.w[:, :, -1] = 0.0
        # Solid stone kills normal flow and drags tangential flow.
        solid = w.stone > 0.5
        w.u[solid] = 0.0
        w.v[solid] = 0.0
        w.w[solid] = 0.0

    # -- steps -----------------------------------------------------------
    def _advect_velocity(self, dt: float) -> None:
        w = self.w
        w.u = _advect(w.u, w.u, w.v, w.w, dt, mode="nearest")
        w.v = _advect(w.v, w.u, w.v, w.w, dt, mode="nearest")
        w.w = _advect(w.w, w.u, w.v, w.w, dt, mode="nearest")

    def _diffuse(self, dt: float) -> None:
        """Implicit viscous diffusion: (I - nu*dt*lap) u_new = u_old."""
        if self.nu <= 0:
            return
        w = self.w
        a = self.nu * dt
        for field in (w.u, w.v, w.w):
            rhs = field.copy()
            for _ in range(self.jacobi_diff):
                field = (rhs + a * (_shift_sum(field))) / (1.0 + 6.0 * a)
            # write back per-component
        # Jacobi above produced locals; redo properly into world arrays:
        for name in ("u", "v", "w"):
            f = getattr(w, name)
            rhs = f.copy()
            for _ in range(self.jacobi_diff):
                f = (rhs + a * _shift_sum(f)) / (1.0 + 6.0 * a)
            setattr(w, name, f.astype(np.float32))

    def _project(self) -> None:
        """Pressure projection to enforce incompressibility."""
        w = self.w
        w.u, w.v, w.w = (np.float32(x) for x in (w.u, w.v, w.w))
        # divergence
        div = np.zeros_like(w.u)
        div[1:] += 0.5 * (w.u[1:] - w.u[:-1])
        div[:-1] += 0.5 * (w.u[1:] - w.u[:-1])
        div[:, 1:] += 0.5 * (w.v[:, 1:] - w.v[:, :-1])
        div[:, :-1] += 0.5 * (w.v[:, 1:] - w.v[:, :-1])
        div[:, :, 1:] += 0.5 * (w.w[:, :, 1:] - w.w[:, :, :-1])
        div[:, :, :-1] += 0.5 * (w.w[:, :, 1:] - w.w[:, :, :-1])
        div *= 0.5

        w.p[:] = 0.0
        for _ in range(self.jacobi_proj):
            w.p = (_shift_sum(w.p) - div) / 6.0

        # subtract pressure gradient
        w.u[1:] -= 0.5 * (w.p[1:] - w.p[:-1])
        w.u[:-1] -= 0.5 * (w.p[1:] - w.p[:-1])
        w.v[:, 1:] -= 0.5 * (w.p[:, 1:] - w.p[:, :-1])
        w.v[:, :-1] -= 0.5 * (w.p[:, 1:] - w.p[:, :-1])
        w.w[:, :, 1:] -= 0.5 * (w.p[:, :, 1:] - w.p[:, :, :-1])
        w.w[:, :, :-1] -= 0.5 * (w.p[:, :, 1:] - w.p[:, :, :-1])

    def step(self, dt: float) -> None:
        self._apply_bounds()
        self._advect_velocity(dt)
        self._apply_bounds()
        self._diffuse(dt)
        self._project()
        self._apply_bounds()


def _shift_sum(f: np.ndarray) -> np.ndarray:
    s = np.zeros_like(f)
    s[1:] += f[:-1]
    s[:-1] += f[1:]
    s[:, 1:] += f[:, :-1]
    s[:, :-1] += f[:, 1:]
    s[:, :, 1:] += f[:, :, :-1]
    s[:, :, :-1] += f[:, :, 1:]
    return s
