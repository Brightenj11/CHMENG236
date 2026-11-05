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


# Pairs named first_second, in beam order. These stay unseen during training. W-first pairs are
# kept out because W stops the protons, which leaves the second cube with little effect.
INITIAL_TEST_PAIRS = {("al", "pb"), ("cu", "al"), ("pb", "cu"), ("cu", "w")}
PAIRS = set(product(MATERIALS, repeat=2))

# dataset: {name: [(i, j, k, material), ...]}
DATASETS = {
    "initial_train": {
        **{f"{m}_center": beam_line(m) for m in MATERIALS},
        **{f"{a}_{b}": beam_line(a, b) for a, b in sorted(PAIRS - INITIAL_TEST_PAIRS)},
    },
    "initial_test": {f"{a}_{b}": beam_line(a, b) for a, b in sorted(INITIAL_TEST_PAIRS)},
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
