"""Streamline tracing through the 3D velocity field.

A streamline is a continuous path that follows the velocity vector at every
point — the classic flow-visualization primitive. We seed the upstream face
and integrate forward (RK4) until the path leaves the domain, runs into stone,
or hits a max length. The result is a set of polylines that bend around the
glyph, showing the river's path and its deflection by the stone.
"""

from __future__ import annotations

import numpy as np


def _sample_vel(field, p, VX, VY, VZ):
    """Trilinear sample of a packed RGBA float32 velocity field at p in [0,1]^3."""
    x = min(VX - 1, max(0.0, p[0] * (VX - 1)))
    y = min(VY - 1, max(0.0, p[1] * (VY - 1)))
    z = min(VZ - 1, max(0.0, p[2] * (VZ - 1)))
    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    stride = 4
    def at(cx, cy, cz):
        cx = min(VX - 1, max(0, cx)); cy = min(VY - 1, max(0, cy)); cz = min(VZ - 1, max(0, cz))
        idx = ((cz * VY + cy) * VX + cx) * stride
        return field[idx], field[idx + 1], field[idx + 2]
    a000 = at(ix, iy, iz); a100 = at(ix + 1, iy, iz)
    a010 = at(ix, iy + 1, iz); a110 = at(ix + 1, iy + 1, iz)
    a001 = at(ix, iy, iz + 1); a101 = at(ix + 1, iy, iz + 1)
    a011 = at(ix, iy + 1, iz + 1); a111 = at(ix + 1, iy + 1, iz + 1)
    def lerp(A, B, t):
        return (A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t)
    x0 = lerp(a000, a100, fx); x1 = lerp(a010, a110, fx)
    x2 = lerp(a001, a101, fx); x3 = lerp(a011, a111, fx)
    y0 = lerp(x0, x1, fy); y1 = lerp(x2, x3, fz)
    return lerp(y0, y1, fz)


def _sample_stone(vox, p, NX, NY, NZ):
    """Trilinear sample of a uint8 stone voxel field; returns [0,1] density."""
    x = min(NX - 1, max(0.0, p[0] * (NX - 1)))
    y = min(NY - 1, max(0.0, p[1] * (NY - 1)))
    z = min(NZ - 1, max(0.0, p[2] * (NZ - 1)))
    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    def c(cx, cy, cz):
        cx = min(NX - 1, max(0, cx)); cy = min(NY - 1, max(0, cy)); cz = min(NZ - 1, max(0, cz))
        return vox[((cz * NY + cy) * NX + cx)] / 255.0
    v = ((1-fx)*(1-fy)*(1-fz)*c(ix,iy,iz) + fx*(1-fy)*(1-fz)*c(ix+1,iy,iz)
         + (1-fx)*fy*(1-fz)*c(ix,iy+1,iz) + fx*fy*(1-fz)*c(ix+1,iy+1,iz)
         + (1-fx)*(1-fy)*fz*c(ix,iy,iz+1) + fx*(1-fy)*fz*c(ix+1,iy,iz+1)
         + (1-fx)*fy*fz*c(ix,iy+1,iz+1) + fx*fy*fz*c(ix+1,iy+1,iz+1))
    return v


def trace_streamlines(
    vel_field: np.ndarray,           # packed RGBA float32 (VX,VY,VZ,4)
    stone_vox: np.ndarray,           # uint8 (NX,NY,NZ)
    vel_grid: tuple[int, int, int],
    grid: tuple[int, int, int],
    n_lines: int = 220,
    max_steps: int = 260,
    step: float = 0.012,
    speed_scale: float = 1.0,
    seed_rng: int = 0,
) -> list[np.ndarray]:
    """Trace ``n_lines`` streamlines; return a list of float32 arrays (Ni,3) in [0,1]^3."""
    VX, VY, VZ = vel_grid
    NX, NY, NZ = grid
    field = np.ascontiguousarray(vel_field).ravel()   # flat RGBA float32
    vox = np.ascontiguousarray(stone_vox).ravel()     # flat uint8 stone
    rng = np.random.default_rng(seed_rng)
    lines: list[np.ndarray] = []

    # seed across the upstream face (x ~ 0) and a few interior columns, spread
    # in y/z so the lines fill the volume and meet the glyph.
    for k in range(n_lines):
        # bias seeds toward the glyph's y/z band so more lines interact with it
        if k % 3 == 0:
            sx = 0.01 + rng.random() * 0.03
            sy = 0.15 + rng.random() * 0.70
            sz = 0.10 + rng.random() * 0.80
        else:
            sx = 0.01 + rng.random() * 0.05
            sy = rng.random()
            sz = rng.random()
        p = [sx, sy, sz]
        pts = [list(p)]
        for _ in range(max_steps):
            v = _sample_vel(field, p, VX, VY, VZ)
            sp = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
            if sp < 1e-4:
                break
            # RK4 step (normalized direction * step, scaled by local speed a bit)
            def deriv(pp):
                vv = _sample_vel(field, pp, VX, VY, VZ)
                nrm = (vv[0]**2 + vv[1]**2 + vv[2]**2) ** 0.5 + 1e-6
                return (vv[0] / nrm, vv[1] / nrm, vv[2] / nrm)
            k1 = deriv(p)
            k2 = deriv([p[0]+step*0.5*k1[0], p[1]+step*0.5*k1[1], p[2]+step*0.5*k1[2]])
            k3 = deriv([p[0]+step*0.5*k2[0], p[1]+step*0.5*k2[1], p[2]+step*0.5*k2[2]])
            k4 = deriv([p[0]+step*k3[0], p[1]+step*k3[1], p[2]+step*k3[2]])
            dx = step * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            dy = step * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            dz = step * (k2[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
            # fix dz (typo-safe): use proper k's
            dz = step * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
            p = [p[0] + dx, p[1] + dy, p[2] + dz]
            if not (0 <= p[0] <= 1 and 0 <= p[1] <= 1 and 0 <= p[2] <= 1):
                break
            if _sample_stone(vox, p, NX, NY, NZ) > 0.5:
                # back off one step so the line stops at the stone surface
                break
            pts.append(list(p))
        if len(pts) >= 4:
            lines.append(np.asarray(pts, dtype=np.float32))
    return lines


def pack_streamlines(lines: list[np.ndarray]) -> dict:
    """Pack a list of polylines into flat arrays for transport."""
    counts = np.array([len(l) for l in lines], dtype=np.uint32)
    flat = np.concatenate(lines).astype(np.float32) if lines else np.zeros((0,), np.float32)
    return {"counts": counts, "points": flat}
