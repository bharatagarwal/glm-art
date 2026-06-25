"""Fluviglyph — typography carved by rivers.

A letterform is dropped into a virtual current. A stable-fluids Navier-Stokes
field flows around it; shear stress ablates the stone; sediment is carried and
re-deposited. After ten thousand iterations the blocky glyph has become
something organic — a mesh extracted by marching cubes.

The river is the sculptor; the code merely watches.
"""

from .word import rasterize_word
from .world import World
from .fluid import FluidSolver
from .erosion import ErosionModel
from .extract import extract_mesh
from .render import (
    save_topdown,
    save_topdown_montage,
    save_cross_section,
    save_mesh_render,
)

__all__ = [
    "rasterize_word",
    "World",
    "FluidSolver",
    "ErosionModel",
    "extract_mesh",
    "save_topdown",
    "save_topdown_montage",
    "save_cross_section",
    "save_mesh_render",
]
