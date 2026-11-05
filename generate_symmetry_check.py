#!/usr/bin/env python3
"""Check in Geant4 that the symmetries are real: rerun a few test geometries rotated and shifted.

Run inside the Nix environment: python generate_symmetry_check.py
Results go to data/symmetry_check/. symmetry_eval.py compares them with the originals.

Each geometry gets three new runs. They all use different random seeds, so their spectra differ by
noise alone. "reseeded" is the same geometry again, which shows how big that noise is.
"""
from pathlib import Path

from dataset import LATTICE_SIZE, read_macro, transform_cubes
from generate_data import ROOT, run

GEOMETRIES = ["line3_al_pb_cu", "offset_corner_al_al_cu", "both_staircase_al_al_cu"]


def variants(cubes):
    depth = [k for _, _, k, _ in cubes]
    z_shift = -2 if min(depth) >= 2 else LATTICE_SIZE - 1 - max(depth)
    return {
        "rotated": transform_cubes(cubes, quarter_turns=1, mirror=True),
        "shifted": transform_cubes(cubes, z_shift=z_shift),
        "reseeded": cubes,
    }


def main():
    (ROOT / "data" / "symmetry_check").mkdir(parents=True, exist_ok=True)
    for name in GEOMETRIES:
        cubes, _ = read_macro(ROOT / "data" / "3_cube" / "test" / f"{name}.mac")
        for variant, moved in variants(cubes).items():
            run("symmetry_check", f"{name}_{variant}", moved)


if __name__ == "__main__":
    main()
