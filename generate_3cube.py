#!/usr/bin/env python3
"""Build data/3_cube/{train,val,test}: every earlier geometry plus 304 three-cube geometries.

Run inside the Nix environment, after generate_data.py has finished:
    python generate_3cube.py
"""
import random
import shutil
from itertools import product

from generate_data import (DATASETS as EARLIER, MATERIALS, ROOT, TEST_LINES, VAL_LINES, beam_line, run)

# (4, 4, 4) then (4, 4, 5) on the beam, with a third cube beside the second one.
OFFSET_POSITION = (5, 4, 5)


def contains(line, pairs):
    """True if any two neighboring cubes along the beam form one of the given pairs."""
    return any(neighbors in pairs for neighbors in zip(line, line[1:]))


def split(line):
    """Keep held-out pairs out of training, so test and validation geometries stay unseen."""
    if contains(line, TEST_LINES):
        return "test"
    return "val" if contains(line, VAL_LINES) else "train"


def behind_and_offset(first, second, offset):
    return beam_line(first, second) + [(*OFFSET_POSITION, MATERIALS[offset])]


NEW = {"train": {}, "val": {}, "test": {}}

# Three cubes in a line along the beam, in every material order: 64 geometries. All are test only:
# the point is to train on shorter stacks and see whether the model can predict a longer one.
for line in product(MATERIALS, repeat=3):
    NEW["test"]["line3_" + "_".join(line)] = beam_line(*line)

# One cube behind another and one offset, with the offset cube in a different material from the
# one beside it. Beam pairs in the test set are skipped, which leaves 12 x 3 = 36 geometries.
for first, second in product(MATERIALS, repeat=2):
    if (first, second) in TEST_LINES:
        continue
    for offset in MATERIALS:
        if offset != second:
            name = f"behind_offset_{first}_{second}_{offset}"
            NEW[split((first, second))][name] = behind_and_offset(first, second, offset)

# The same shape with the offset cube in other places. Each layout uses the 12 allowed beam pairs,
# with the offset material rotating through the 3 options, so there are 12 geometries per layout.
OFFSET_LAYOUTS = {
    "out": (6, 4, 5),
    "far": (8, 4, 5),
    "corner": (5, 5, 5),  # test only, so this position is never seen in training
    "behind": (5, 4, 6),
    "beside_first": (5, 4, 4),
}
ALLOWED_PAIRS = sorted(set(product(MATERIALS, repeat=2)) - TEST_LINES)
for layout, position in OFFSET_LAYOUTS.items():
    for index, (first, second) in enumerate(ALLOWED_PAIRS):
        offset = [m for m in MATERIALS if m != second][index % 3]
        name = f"offset_{layout}_{first}_{second}_{offset}"
        target = "test" if layout == "corner" else split((first, second))
        NEW[target][name] = beam_line(first, second) + [(*position, MATERIALS[offset])]

# One cube on the beam and both others offset, by the same or different amounts. Each layout uses
# 24 of the 48 combinations where the offset cubes differ in material, picked with a fixed seed.
BOTH_OFFSET_LAYOUTS = {
    "same_side": [(5, 4, 5), (5, 4, 6)],
    "opposite": [(5, 4, 5), (3, 4, 5)],
    "row": [(5, 4, 5), (6, 4, 5)],
    "staircase": [(5, 4, 5), (6, 4, 6)],  # test only, so this layout is never seen in training
    "mixed_amounts": [(5, 4, 5), (3, 3, 6)],
    "wide": [(6, 4, 5), (2, 4, 6)],
}
BOTH_OFFSET_VAL = 3  # geometries per training layout held out for validation
COMBINATIONS = [combo for combo in product(MATERIALS, repeat=3) if combo[1] != combo[2]]
for layout, positions in BOTH_OFFSET_LAYOUTS.items():
    for index, combo in enumerate(sorted(random.Random(layout).sample(COMBINATIONS, 24))):
        on_axis, *offsets = combo
        target = "test" if layout == "staircase" else "val" if index < BOTH_OFFSET_VAL else "train"
        NEW[target][f"both_{layout}_{'_'.join(combo)}"] = [(4, 4, 4, MATERIALS[on_axis])] + [
            (*position, MATERIALS[m]) for position, m in zip(positions, offsets)
        ]


def main():
    # Lines simulated before they became test only. Moving them keeps them out of training.
    test_dir = ROOT / "data" / "3_cube" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        for path in (ROOT / "data" / "3_cube" / split_name).glob("line3_*"):
            shutil.move(path, test_dir / path.name)
    for earlier, split_name in (("initial_train", "train"), ("initial_val", "val"), ("initial_test", "test")):
        target = ROOT / "data" / "3_cube" / split_name
        target.mkdir(parents=True, exist_ok=True)
        for name in EARLIER[earlier]:
            # Same physics and seed as a rerun, so copy the finished result instead of repeating it.
            for suffix in (".mac", "_nt_hits_t0.csv"):
                shutil.copy2(ROOT / "data" / earlier / f"{name}{suffix}", target / f"{name}{suffix}")
        print(f"3_cube/{split_name}: copied {len(EARLIER[earlier])} earlier geometries", flush=True)
    for split_name, geometries in NEW.items():
        for name, cubes in geometries.items():
            run(f"3_cube/{split_name}", name, cubes)


if __name__ == "__main__":
    main()
