"""Diagnose why erosion isn't biting."""
import numpy as np
from fluviglyph import World, FluidSolver, ErosionModel
from fluviglyph.erosion import _air_adjacent_to_stone

w = World.from_word("WORN", nx=64, ny=32, nz=28)
fluid = FluidSolver(w, viscosity=1e-5)
ero = ErosionModel(w, dt=0.25)

print("initial stone mass:", w.stone.sum())
surf0 = _air_adjacent_to_stone(w.stone)
print("air-attack voxels:", int(surf0.sum()))

for it in range(5):
    fluid.step(0.25)
    speed = np.sqrt(w.u**2 + w.v**2 + w.w**2)
    surf = _air_adjacent_to_stone(w.stone)
    print(f"\niter {it+1}")
    print(f"  speed: max={speed.max():.3f}  mean_on_attack={speed[surf].mean():.4f}  max_on_attack={speed[surf].max():.4f}")
    pre = w.stone.sum()
    ero.step()
    post = w.stone.sum()
    print(f"  stone mass: {pre:.1f} -> {post:.1f}  (delta {post-pre:.4f})")
    print(f"  sediment max: {w.sediment.max():.5f}")
