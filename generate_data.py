#!/usr/bin/env python3
"""Run Geant4 for sets of cube geometries and save the results under data/.

Run inside the Nix environment, after building:
    python generate_data.py

For each geometry this writes, in data/<dataset>/:
    <name>.mac               the exact run settings, including the cube list
    <name>_nt_hits_t0.csv    protons and neutrons that reached the detector
"""
import subprocess
import zlib
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent
PRIMARIES = 100000
MATERIALS = {"al": "G4_Al", "cu": "G4_Cu", "w": "G4_W", "pb": "G4_Pb"}


def beam_line(*materials):
    """Touching cubes along the beam, in the order the beam hits them, starting at the center."""
    # Positions run from 0 to 8, and (4, 4, *) is on the beam axis.
    return [(4, 4, 4 + offset, MATERIALS[m]) for offset, m in enumerate(materials)]


def beside(on_axis, off_axis):
    """On-axis cube with a touching cube next to it, outside the beam."""
    return [(4, 4, 4, MATERIALS[on_axis]), (5, 4, 4, MATERIALS[off_axis])]


def diagonal(on_axis, off_axis):
    """On-axis cube with a cube diagonally behind it, reached only by scattered particles."""
    return [(4, 4, 4, MATERIALS[on_axis]), (5, 4, 5, MATERIALS[off_axis])]


def on_axis_plus(on_axis, off_axis, *positions):
    """On-axis cube at the center, plus off-axis cubes of one material at the given positions."""
    return [(4, 4, 4, MATERIALS[on_axis])] + [(*position, MATERIALS[off_axis]) for position in positions]


def missed(material, *positions):
    """Cubes outside the beam, with nothing on the axis. The beam should pass straight by."""
    assert all(position[:2] != (4, 4) for position in positions)
    return [(*position, MATERIALS[material]) for position in positions]


PAIRS = set(product(MATERIALS, repeat=2))

# Off-axis cubes that an on-axis cube can scatter particles into: name: positions
OFF_AXIS_LAYOUTS = {
    "diagonal_corner": [(5, 5, 5)],
    "diagonal_far": [(5, 4, 6)],
    "diagonal_out": [(6, 4, 5)],
    "beside_corner": [(5, 5, 4)],
    "two_behind": [(5, 4, 5), (3, 4, 5)],
}

# Beam misses every cube. Distances and depths vary, and directions are one-sided because the
# others are rotations of these around the beam. "missed" keeps the name of the first runs.
MISSED_LAYOUTS = {
    "missed": [(5, 4, 4)],
    "missed_out": [(6, 4, 4)],
    "missed_edge": [(8, 4, 4)],
    "missed_corner": [(5, 5, 4)],
    "missed_far_corner": [(6, 6, 4)],
    "missed_lattice_corner": [(8, 8, 4)],
    "missed_front": [(5, 4, 1)],
    "missed_back": [(5, 4, 7)],
    "missed2_line": [(5, 4, 4), (5, 4, 5)],
    "missed2_both_sides": [(5, 4, 4), (3, 4, 4)],
    "missed2_opposite_corners": [(5, 5, 4), (3, 3, 4)],
    "missed2_row": [(5, 4, 4), (6, 4, 4)],
    "missed2_separated": [(5, 4, 2), (5, 4, 7)],
    "missed2_wide": [(6, 4, 4), (2, 4, 4)],
    "missed2_corner_line": [(5, 5, 4), (5, 5, 5)],
    "missed2_far_corners": [(8, 8, 0), (0, 0, 8)],
}
VAL_OFF_AXIS = {("cu", "pb"), ("al", "w")}  # held out of each layout in OFF_AXIS_LAYOUTS
VAL_MISSED = {"missed_edge_cu", "missed_back_w", "missed2_both_sides_pb", "missed2_row_al"}

OFF_AXIS = {f"{layout}_{a}_{b}": on_axis_plus(a, b, *positions)
            for layout, positions in OFF_AXIS_LAYOUTS.items() for a, b in sorted(PAIRS)}
MISSED = {f"{layout}_{m}": missed(m, *positions)
          for layout, positions in MISSED_LAYOUTS.items() for m in MATERIALS}
VAL_OFF_AXIS_NAMES = {f"{layout}_{a}_{b}" for layout in OFF_AXIS_LAYOUTS for a, b in VAL_OFF_AXIS}

# Beam-line pairs named first_second, in beam order. These stay unseen during training. W-first
# pairs are kept out because W stops the protons, which leaves the second cube with little effect.
TEST_LINES = {("al", "pb"), ("cu", "al"), ("pb", "cu"), ("cu", "w")}

# Held out of training for tuning and the loss curve, so the test set stays untouched.
VAL_LINES = {("al", "cu"), ("pb", "w")}
VAL_BESIDE = {("cu", "pb"), ("w", "al")}
VAL_DIAGONAL = {("al", "w"), ("pb", "cu")}

# dataset: {name: [(i, j, k, material), ...]}
DATASETS = {
    "initial_train": {
        **{f"{m}_center": beam_line(m) for m in MATERIALS},
        **{f"{a}_{b}": beam_line(a, b) for a, b in sorted(PAIRS - TEST_LINES - VAL_LINES)},
        **{f"beside_{a}_{b}": beside(a, b) for a, b in sorted(PAIRS - VAL_BESIDE)},
        **{f"diagonal_{a}_{b}": diagonal(a, b) for a, b in sorted(PAIRS - VAL_DIAGONAL)},
        **{name: cubes for name, cubes in OFF_AXIS.items() if name not in VAL_OFF_AXIS_NAMES},
        **{name: cubes for name, cubes in MISSED.items() if name not in VAL_MISSED},
    },
    "initial_val": {
        **{f"{a}_{b}": beam_line(a, b) for a, b in sorted(VAL_LINES)},
        **{f"beside_{a}_{b}": beside(a, b) for a, b in sorted(VAL_BESIDE)},
        **{f"diagonal_{a}_{b}": diagonal(a, b) for a, b in sorted(VAL_DIAGONAL)},
        **{name: OFF_AXIS[name] for name in sorted(VAL_OFF_AXIS_NAMES)},
        **{name: MISSED[name] for name in sorted(VAL_MISSED)},
    },
    "initial_test": {f"{a}_{b}": beam_line(a, b) for a, b in sorted(TEST_LINES)},
}

MACRO = """\
{cubes}
/run/initialize
/analysis/setFileName {stem}
/random/setSeeds {seed} {seed2}
/run/verbose 1
/event/verbose 0
/tracking/verbose 0
/run/printProgress 10000
/gun/particle proton
/gun/energy 100 MeV
/run/beamOn {primaries}
"""


def run(dataset, name, cubes):
    stem = Path("data") / dataset / name
    if (ROOT / f"{stem}_nt_hits_t0.csv").exists():  # delete data/ after changing the simulation
        print(f"{dataset}/{name}: already done")
        return
    # The seed comes from the name, so each geometry gets its own stream and reruns are repeatable.
    seed = zlib.crc32(name.encode()) % 2**31
    macro = stem.with_suffix(".mac")
    (ROOT / macro).write_text(MACRO.format(
        stem=stem,
        cubes="\n".join(f"/geom/addCube {i} {j} {k} {material}" for i, j, k, material in cubes),
        seed=seed,
        seed2=(seed // 2) + 1,
        primaries=PRIMARIES,
    ))
    result = subprocess.run([ROOT / "build" / "exampleB1", macro], cwd=ROOT,
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    output = ROOT / f"{stem}_nt_hits_t0.csv"
    if result.returncode != 0 or not output.exists():
        raise SystemExit(f"{name} failed:\n{log[-3000:]}")
    if "GeomVol1002" in log:  # Geant4's overlap warning: the geometry isn't what was asked for
        raise SystemExit(f"{name} has overlapping volumes:\n{log[-3000:]}")
    print(f"{dataset}/{name}: done")


def main():
    for dataset, geometries in DATASETS.items():
        (ROOT / "data" / dataset).mkdir(parents=True, exist_ok=True)
        for name, cubes in geometries.items():
            run(dataset, name, cubes)


if __name__ == "__main__":
    main()
