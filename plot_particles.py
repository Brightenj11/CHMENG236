#!/usr/bin/env python3
"""Plot the energy spectrum and angular distribution from a Geant4 detector CSV.

Usage:
    python plot_particles.py data/initial_train/al_al_nt_hits_t0.csv --primaries 100000
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, so this also works over ssh and in nix shells
import matplotlib.pyplot as plt

from dataset import ANGLE_EDGES, ENERGY_EDGES, PARTICLES, load_hits, spectra

COLORS = {"proton": "#2a78d6", "neutron": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def style(ax, xlabel, ylabel, title):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")


def draw(ax, yields, edges, labels=None):
    """Draw one step line per particle. Empty bins become gaps, since log axes can't show zero."""
    for index, name in enumerate(PARTICLES):
        values = yields[index].astype(float)
        values[values <= 0] = float("nan")
        ax.stairs(values, edges, color=COLORS[name], linewidth=1.5,
                  label=labels[name] if labels else name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--primaries", type=int, default=100000,
                        help="protons fired: must match /run/beamOn in the macro (default: 100000)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output image (default: plots/<csv folder>/<csv name>.png)")
    args = parser.parse_args()
    if args.primaries <= 0:
        parser.error("--primaries must be positive")

    hits = load_hits(args.csv)
    if not hits:
        raise SystemExit(f"No particle rows in {args.csv}")
    energy, angle = spectra(hits, args.primaries)
    labels = {p: f"{p} ({len(hits[p][0]) if p in hits else 0:,} detected)" for p in PARTICLES}

    fig, (energy_ax, angle_ax) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    draw(energy_ax, energy, ENERGY_EDGES, labels)
    style(energy_ax, "Kinetic energy (MeV)", "Particles per incident proton per bin", "Energy spectrum")
    energy_ax.set_xscale("log")
    energy_ax.set_xlim(ENERGY_EDGES[0], ENERGY_EDGES[-1])
    draw(angle_ax, angle, ANGLE_EDGES)
    style(angle_ax, "Angle from beam axis, θ (degrees)", "Particles per incident proton per bin",
          "Angular distribution")
    angle_ax.set_xlim(ANGLE_EDGES[0], ANGLE_EDGES[-1])

    handles, legend_labels = energy_ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper left", ncols=2, frameon=False,
               labelcolor=INK)

    output = args.output or Path(__file__).parent / "plots" / args.csv.parent.name / f"{args.csv.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="#fcfcfb")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
