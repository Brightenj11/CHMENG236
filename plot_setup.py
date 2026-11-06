#!/usr/bin/env python3
"""Figure of the simulation setup: (a) the whole world to scale, (b) the cube lattice with an example.

Usage: python plot_setup.py [geometry .mac]   (default: the unseen staircase test geometry)
Output: plots/results/setup.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from dataset import LATTICE_SIZE, read_macro
from plot_particles import COLORS, GRID, INK, MUTED

ROOT = Path(__file__).parent
# Must match src/: detector radius, gun position, lattice half width (all cm)
DETECTOR_R, GUN_Z, HALF = 130, -50, LATTICE_SIZE / 2
MATERIAL_COLORS = {"G4_Al": "#c9ccd1", "G4_Cu": "#c27a3e", "G4_W": "#4a4f57", "G4_Pb": "#7d8a9c",
                   "G4_Ag": "#e6e6ea"}
BEAM = COLORS["proton"]


def world(ax):
    """Side view (x against z) to scale."""
    ax.add_patch(Circle((0, 0), DETECTOR_R, fill=False, color=INK, linewidth=2.5))
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, color=MATERIAL_COLORS["G4_Cu"]))
    ax.annotate("", xy=(-HALF - 1, 0), xytext=(GUN_Z, 0),
                arrowprops=dict(arrowstyle="-|>", color=BEAM, linewidth=2))
    ax.plot(GUN_Z, 0, "o", color=BEAM, markersize=6)
    ax.text(GUN_Z, -9, "100 MeV proton gun\nz = -50 cm", ha="center", va="top", color=BEAM, fontsize=9)
    ax.text(-3, 7, "cube lattice\n9 x 9 x 9 cm", ha="right", va="bottom", color=MUTED, fontsize=9)
    ax.annotate("spherical detector, r = 1.3 m (4π)", xy=(-92, 92), xytext=(-140, 145),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))
    # Scattering angle theta, measured from the beam axis, 0 to 180 degrees.
    theta = np.radians(40)
    ax.add_patch(FancyArrowPatch((0, 0), (DETECTOR_R * np.cos(theta), DETECTOR_R * np.sin(theta)),
                                 arrowstyle="-|>", mutation_scale=14, color=COLORS["neutron"],
                                 linewidth=1.5))
    ax.plot([0, DETECTOR_R], [0, 0], color=MUTED, linewidth=0.8, linestyle=":")
    ax.add_patch(Arc((0, 0), 70, 70, theta1=0, theta2=40, color=COLORS["neutron"]))
    ax.text(40, 13, "θ", color=COLORS["neutron"], fontsize=12)
    ax.text(DETECTOR_R * 0.80, DETECTOR_R * 0.70, "outgoing p, n:\nenergy and θ", color=COLORS["neutron"],
            fontsize=9)
    ax.set_xlim(-145, 145)
    ax.set_ylim(-145, 160)
    ax.set_aspect("equal")
    ax.set_xlabel("z, beam axis (cm)", color=MUTED)
    ax.set_ylabel("x (cm)", color=MUTED)
    ax.set_title("(a) Geant4 world, to scale", color=INK, loc="left", fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)


def lattice(ax, cubes, title):
    """3D view of the lattice. Plot axes are (z, x, y) so the beam runs left to right."""
    filled = np.zeros((LATTICE_SIZE,) * 3, bool)
    colors = np.empty(filled.shape, object)
    for i, j, k, material in cubes:
        filled[k, i, j] = True
        colors[k, i, j] = MATERIAL_COLORS[material]
    ax.voxels(filled, facecolors=colors, edgecolor="#2b2b2b", linewidth=0.5)
    # Lattice outline
    n = LATTICE_SIZE
    for a in (0, n):
        for b in (0, n):
            for line in (([0, n], [a, a], [b, b]), ([a, a], [0, n], [b, b]), ([a, a], [b, b], [0, n])):
                ax.plot(*line, color=GRID, linewidth=0.8)
    center = n / 2
    front = min(k for i, j, k, _ in cubes if (i, j) == (n // 2, n // 2)) if any(
        (i, j) == (n // 2, n // 2) for i, j, *_ in cubes) else n
    ax.plot([front, n], [center, center], [center, center], color=BEAM, linewidth=1, linestyle=":")
    ax.quiver(-4, center, center, front + 4, 0, 0, color=BEAM, linewidth=2.5, arrow_length_ratio=0.08)
    ax.text(-4, center, center + 1.0, "beam", color=BEAM, fontsize=9)
    ax.set_xlim(-4, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((n + 4, n, n))
    ax.set_xlabel("z (cm)", color=MUTED)
    ax.set_ylabel("x (cm)", color=MUTED)
    ax.set_zlabel("y (cm)", color=MUTED)
    ax.view_init(elev=48, azim=-72)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    shown = sorted({m for *_, m in cubes})
    handles = [Rectangle((0, 0), 1, 1, facecolor=MATERIAL_COLORS[m], edgecolor="#2b2b2b") for m in shown]
    ax.legend(handles, [m.removeprefix("G4_") for m in shown], frameon=False, loc="upper right")


def main():
    macro = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "data" / "3_cube" / "test" / "both_staircase_al_al_cu.mac")
    cubes, _ = read_macro(macro)
    fig = plt.figure(figsize=(12, 5.2))
    world(fig.add_subplot(1, 2, 1))
    lattice(fig.add_subplot(1, 2, 2, projection="3d"), cubes,
            "(b) 9 x 9 x 9 lattice of 1 cm cube slots\n     example test geometry")
    fig.tight_layout()
    out = ROOT / "plots" / "results" / "setup.png"
    fig.savefig(out, dpi=200)
    print(f"Plot: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
