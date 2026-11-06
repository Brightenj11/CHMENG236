#!/usr/bin/env python3
"""Unseen-material test: one silver cube (Z 47, 10.5 g/cm^3) on the beam axis.

Training only has Al (13), Cu (29), W (74) and Pb (82), so silver tests whether the model can fill in
the gap between Cu and W. The data comes from generate_data.run("unseen_material", "ag_center", ...).

Usage: python unseen_material.py
Baselines: copy the Cu or W spectrum, or interpolate the log spectra between Cu and W by Z.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import ANGLE_EDGES, EMPTY_BIN_YIELD, ENERGY_EDGES, PARTICLES, load_hits, read_macro, spectra, voxelize
from models import VoxelCNN
from plot_particles import COLORS, style
from symmetry import canonicalize
from train import to_log

ROOT = Path(__file__).parent
MODELS = {"3_cube": "none", **{f"sym_augment_s{s}": "augment" for s in range(3)},
          **{f"sym_canonical_s{s}": "canonical" for s in range(3)}}


def load(stem):
    cubes, primaries = read_macro(f"{stem}.mac")
    energy, angle = spectra(load_hits(f"{stem}_nt_hits_t0.csv"), primaries)
    return cubes, to_log(energy), to_log(angle)


def errors(pe, pa, te, ta):
    """(proton energy, proton angle, neutron energy, neutron angle) log RMSE, then total yields
    relative to the truth (from the energy output)."""
    e, a = ((pe - te) ** 2).mean(-1).sqrt(), ((pa - ta) ** 2).mean(-1).sqrt()
    total = (pe.exp() - EMPTY_BIN_YIELD).sum(-1) / (te.exp() - EMPTY_BIN_YIELD).sum(-1) - 1
    return [e[0], a[0], e[1], a[1], total[0], total[1]]


def main():
    cubes, true_e, true_a = load(ROOT / "data" / "unseen_material" / "ag_center")
    _, cu_e, cu_a = load(ROOT / "data" / "3_cube" / "train" / "cu_center")
    _, w_e, w_a = load(ROOT / "data" / "3_cube" / "train" / "w_center")
    t = (47 - 29) / (74 - 29)  # silver's place between Cu and W in Z
    rows = {"copy Cu": (cu_e, cu_a), "copy W": (w_e, w_a),
            "Z interpolation Cu-W": ((1 - t) * cu_e + t * w_e, (1 - t) * cu_a + t * w_a)}

    voxels = torch.as_tensor(voxelize(cubes))[None]
    for name, method in MODELS.items():
        model = VoxelCNN()
        model.load_state_dict(torch.load(ROOT / "results" / f"{name}_voxel_cnn.pt"))
        model.eval()
        with torch.no_grad():
            pe, pa = model(canonicalize(voxels) if method == "canonical" else voxels)
        rows[name] = (pe[0], pa[0])

    print("Single silver cube. Log-space RMSE per bin, and total yield vs Geant4 (from energy output).")
    print(f"  {'prediction':22s} {'p energy':>9s} {'p angle':>8s} {'n energy':>9s} {'n angle':>8s}"
          f" {'p total':>8s} {'n total':>8s}")
    for name, (pe, pa) in rows.items():
        v = errors(pe, pa, true_e, true_a)
        print(f"  {name:22s} {v[0]:9.3f} {v[1]:8.3f} {v[2]:9.3f} {v[3]:8.3f} {v[4]:+8.1%} {v[5]:+8.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pe, pa = rows["3_cube"]
    for ax, true, pred, edges, title, xlabel in (
            (axes[0], true_e, pe, ENERGY_EDGES, "Energy", "Energy (MeV)"),
            (axes[1], true_a, pa, ANGLE_EDGES, "Angle", "Angle from beam (degrees)")):
        for index, particle in enumerate(PARTICLES):
            for values, line, label in ((true, "-", "Geant4"), (pred, "--", "3_cube model")):
                y = (values[index].exp() - EMPTY_BIN_YIELD).numpy().astype(float)
                y[y <= 0.5 * EMPTY_BIN_YIELD] = np.nan
                ax.stairs(y, edges, color=COLORS[particle], linestyle=line, linewidth=1.5,
                          label=f"{particle}, {label}")
        style(ax, xlabel, "Yield per incident proton per bin", f"{title}: single silver cube (unseen)")
    axes[0].set_xscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = ROOT / "plots" / "results" / "unseen_material_ag.png"
    fig.savefig(out, dpi=150)
    print(f"Plot: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
