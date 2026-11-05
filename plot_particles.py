#!/usr/bin/env python3
"""Plot the energy spectrum and angular distribution from a Geant4 detector CSV.

Usage:
    python plot_particles.py build/particles_nt_hits_t0.csv --primaries 100000
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, so this also works over ssh and in nix shells
import matplotlib.pyplot as plt
import numpy as np

# Fixed bins so that plots from different cubes can be compared directly.
ENERGY_BINS = np.arange(0, 101, 1)  # MeV
THETA_BINS = np.arange(0, 91, 1)  # degrees from the beam axis

COLORS = {"proton": "#2a78d6", "neutron": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load(path):
    """Return {particle: (energies in MeV, thetas in degrees)}. Lines starting with # are header."""
    columns = {}
    with open(path, newline="") as f:
        for _event, particle, energy, theta, _phi in csv.reader(
            line for line in f if not line.startswith("#")
        ):
            energies, thetas = columns.setdefault(particle, ([], []))
            energies.append(float(energy))
            thetas.append(float(theta))
    return {name: (np.array(e), np.array(t)) for name, (e, t) in columns.items()}


def draw(ax, data, column, bins, primaries, xlabel, ylabel, title):
    width = np.diff(bins)
    for name, color in COLORS.items():
        if name not in data:
            continue
        counts, _ = np.histogram(data[name][column], bins=bins)
        # Normalize by protons fired, not rows: events with no detected particle leave no row.
        yields = counts / (primaries * width)
        # Empty bins become gaps, since zero can't be drawn on a log axis.
        ax.stairs(np.where(counts > 0, yields, np.nan), bins, color=color, linewidth=1.5,
                  label=f"{name} ({len(data[name][column]):,} detected)")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default="build/particles_nt_hits_t0.csv", type=Path)
    parser.add_argument("--primaries", type=int, default=100000,
                        help="protons fired: must match /run/beamOn in the macro (default: 100000)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output image (default: plots/<csv name>.png in the project folder)")
    args = parser.parse_args()
    if args.primaries <= 0:
        parser.error("--primaries must be positive")

    data = load(args.csv)
    if not data:
        raise SystemExit(f"No particle rows in {args.csv}")

    fig, (energy_ax, theta_ax) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    draw(energy_ax, data, 0, ENERGY_BINS, args.primaries, "Kinetic energy (MeV)",
         "Particles per incident proton per MeV", "Energy spectrum")
    draw(theta_ax, data, 1, THETA_BINS, args.primaries, "Angle from beam axis, θ (degrees)",
         "Particles per incident proton per degree", "Angular distribution")

    handles, labels = energy_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper left", ncols=2, frameon=False,
               labelcolor=INK)

    output = args.output or Path(__file__).parent / "plots" / f"{args.csv.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="#fcfcfb")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
