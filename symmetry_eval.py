#!/usr/bin/env python3
"""Compare how well each model respects the symmetries, on the 3_cube test set.

Usage: python symmetry_eval.py

For every test geometry it builds up to 24 copies: 8 rotations and mirrors around the beam, each at
up to 3 positions along it (moved to the front, unmoved, and moved to the back). They all have the
same correct spectra, so no new simulations are needed. For each model it reports:
- accuracy on the original test set
- accuracy on all the moved copies
- consistency: how much the predictions for copies of one geometry differ from each other.
  0 means exactly symmetric.
It also compares Geant4 runs of a few rotated and shifted geometries, from generate_symmetry_check.py.
"""
from pathlib import Path

import numpy as np
import torch

from dataset import PARTICLES, load_hits, read_macro, spectra, load_dataset
from models import VoxelCNN
from symmetry import ROTATIONS, canonicalize, rotate, shift_z, z_shift_range
from train import test_group, to_log

ROOT = Path(__file__).parent
SEEDS = (0, 1, 2)
# label: (checkpoint names, symmetry method)
MODELS = {
    "No symmetry (current 3_cube model)": (["3_cube"], "none"),
    "Augmentation": ([f"sym_augment_s{s}" for s in SEEDS], "augment"),
    "Canonicalization": ([f"sym_canonical_s{s}" for s in SEEDS], "canonical"),
}


def moved_copies(sample):
    """All rotated, mirrored and shifted copies of one (2, x, y, z) geometry."""
    copies = []
    for turns, mirror in ROTATIONS:
        rotated = rotate(sample[None], turns, mirror)[0]
        low, high = z_shift_range(rotated)
        for shift in sorted({low, 0, high}):
            copies.append(shift_z(rotated, shift))
    return torch.stack(copies)


def predict(model, voxels, method):
    inputs = canonicalize(voxels) if method == "canonical" else voxels
    with torch.no_grad():
        return model(inputs)


def errors(pred_e, pred_a, true_e, true_a):
    """Log-space RMSE per bin for (proton energy, proton angle, neutron energy, neutron angle)."""
    e = ((pred_e - true_e) ** 2).mean(dim=(0, 2)).sqrt()
    a = ((pred_a - true_a) ** 2).mean(dim=(0, 2)).sqrt()
    return np.array([e[0], a[0], e[1], a[1]])


def spread(pred_e, pred_a):
    """RMS difference of each copy's prediction from the copies' average, per particle and output."""
    e = ((pred_e - pred_e.mean(dim=0)) ** 2).mean(dim=(0, 2)).sqrt()
    a = ((pred_a - pred_a.mean(dim=0)) ** 2).mean(dim=(0, 2)).sqrt()
    return np.array([e[0], a[0], e[1], a[1]])


def evaluate(checkpoint, method, names, voxels, true_e, true_a, copies):
    model = VoxelCNN()
    model.load_state_dict(torch.load(ROOT / "results" / f"{checkpoint}_voxel_cnn.pt"))
    model.eval()
    groups = np.array([test_group(n) for n in names])
    rows = {}
    original = predict(model, voxels, method)
    moved_e, moved_a, spreads = [], [], []
    for index, sample_copies in enumerate(copies):
        pe, pa = predict(model, sample_copies, method)
        moved_e.append(errors(pe, pa, true_e[index:index + 1], true_a[index:index + 1]))
        spreads.append(spread(pe, pa))
    moved_e, spreads = np.array(moved_e), np.array(spreads)
    for group in ["all"] + sorted(set(groups)):
        members = np.ones(len(names), bool) if group == "all" else groups == group
        mask = torch.as_tensor(members)
        rows[group] = {
            "original": errors(original[0][mask], original[1][mask], true_e[mask], true_a[mask]),
            # Squared errors average across geometries, then the root is taken.
            "moved": np.sqrt((moved_e[members] ** 2).mean(axis=0)),
            "consistency": np.sqrt((spreads[members] ** 2).mean(axis=0)),
        }
    return rows


def geant4_check():
    print("\nGeant4 check: neutron spectra of moved geometries vs the original (log-space RMSE per bin)")
    print("Rotated and shifted should match reseeded, which is the same geometry with new random numbers.")
    check = ROOT / "data" / "symmetry_check"
    if not check.exists():
        print("  (no data/symmetry_check yet)")
        return
    neutron = PARTICLES.index("neutron")
    for original in sorted({p.name.rsplit("_", 1)[0] for p in check.glob("*.mac")}):
        base = ROOT / "data" / "3_cube" / "test" / original
        _, primaries = read_macro(f"{base}.mac")
        base_e, base_a = spectra(load_hits(f"{base}_nt_hits_t0.csv"), primaries)
        line = f"  {original:26s}"
        for variant in ("rotated", "shifted", "reseeded"):
            stem = check / f"{original}_{variant}"
            _, primaries = read_macro(f"{stem}.mac")
            e, a = spectra(load_hits(f"{stem}_nt_hits_t0.csv"), primaries)
            difference = np.sqrt(np.mean(np.concatenate([
                (to_log(e[neutron]) - to_log(base_e[neutron])).numpy() ** 2,
                (to_log(a[neutron]) - to_log(base_a[neutron])).numpy() ** 2])))
            total = e[neutron].sum() / base_e[neutron].sum() - 1
            line += f"  {variant} {difference:.3f} (total {total:+.1%})"
        print(line)


def main():
    names, voxels, energy, angle = load_dataset(ROOT / "data" / "3_cube" / "test")
    voxels = torch.as_tensor(voxels)
    true_e, true_a = to_log(energy), to_log(angle)
    copies = [moved_copies(sample) for sample in voxels]
    print(f"{len(names)} test geometries, {sum(len(c) for c in copies)} moved copies")

    columns = ("proton energy", "proton angle", "neutron energy", "neutron angle")
    for label, (checkpoints, method) in MODELS.items():
        found = [c for c in checkpoints if (ROOT / "results" / f"{c}_voxel_cnn.pt").exists()]
        if not found:
            print(f"\n{label}: no trained models yet")
            continue
        runs = [evaluate(c, method, names, voxels, true_e, true_a, copies) for c in found]
        print(f"\n{label}: {len(found)} seed(s), mean ± spread across seeds")
        print(f"  {'group':26s} {'measure':12s} " + "  ".join(f"{c:>15s}" for c in columns))
        for group in runs[0]:
            for measure in ("original", "moved", "consistency"):
                values = np.array([run[group][measure] for run in runs])
                cells = [f"{m:6.3f} ± {s:5.3f}" for m, s in zip(values.mean(axis=0), values.std(axis=0))]
                print(f"  {group:26s} {measure:12s} " + "  ".join(f"{c:>15s}" for c in cells))
    geant4_check()


if __name__ == "__main__":
    main()
