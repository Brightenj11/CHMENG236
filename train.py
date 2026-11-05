#!/usr/bin/env python3
"""Train VoxelCNN on one dataset and evaluate it on geometries it has never seen.

Usage:
    python train.py
    python train.py --train-dir data/initial_train --test-dir data/initial_test --epochs 300
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dataset import ANGLE_EDGES, EMPTY_BIN_YIELD, ENERGY_EDGES, PARTICLES, load_dataset
from models import VoxelCNN
from plot_particles import COLORS, draw, style

ROOT = Path(__file__).parent


def to_log(yields):
    return torch.log(torch.as_tensor(yields) + EMPTY_BIN_YIELD)


def rmse(pred, true):
    """Log-space error for each particle. 0.69 means off by about a factor of 2 per bin."""
    return ((pred - true) ** 2).mean(dim=(0, 2)).sqrt()


def report(label, energy, angle, true_energy, true_angle):
    energy_error, angle_error = rmse(energy, true_energy), rmse(angle, true_angle)
    for index, particle in enumerate(PARTICLES):
        print(f"  {label:9s} {particle:8s} energy {energy_error[index]:.3f}   angle {angle_error[index]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data" / "initial_train")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "initial_test")
    # Fixed length: stopping when the test error looks best would leak the test set into training.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    # CPU by default: on an M1, the macOS GPU watchdog killed training on "mps" with a batch of 16.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--angle-weight", type=float, default=1.0,
                        help="loss = energy error + angle-weight * angle error")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    train_names, train_voxels, train_energy, train_angle = load_dataset(args.train_dir)
    test_names, test_voxels, test_energy, test_angle = load_dataset(args.test_dir)
    if set(train_names) & set(test_names):
        raise SystemExit(f"Same geometry in train and test: {set(train_names) & set(test_names)}")
    print(f"{len(train_names)} train and {len(test_names)} test geometries on {device}")

    train_x = torch.as_tensor(train_voxels, device=device)
    test_x = torch.as_tensor(test_voxels, device=device)
    train_e, train_a = to_log(train_energy).to(device), to_log(train_angle).to(device)
    test_e, test_a = to_log(test_energy).to(device), to_log(test_angle).to(device)

    # Baseline: always predict the average training spectrum.
    mean_e, mean_a = train_e.mean(dim=0), train_a.mean(dim=0)

    model = VoxelCNN().to(device)
    # Start from the baseline, so the network only has to learn how each geometry differs from it.
    with torch.no_grad():
        model.energy_head.bias.copy_(mean_e.flatten())
        model.angle_head.bias.copy_(mean_a.flatten())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in torch.randperm(len(train_x), device=device).split(args.batch_size):
            pred_e, pred_a = model(train_x[batch])
            loss = (nn.functional.mse_loss(pred_e, train_e[batch])
                    + args.angle_weight * nn.functional.mse_loss(pred_a, train_a[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:4d}   train loss {np.mean(losses):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        final_e, final_a = model(train_x)
        pred_e, pred_a = model(test_x)

    print("\nLog-space RMSE per bin (lower is better, 0.69 is about a factor of 2):")
    report("train", final_e, final_a, train_e, train_a)
    report("test", pred_e, pred_a, test_e, test_a)
    report("baseline", mean_e.expand_as(test_e), mean_a.expand_as(test_a), test_e, test_a)

    neutron = PARTICLES.index("neutron")
    print("\nNeutrons per incident proton, test geometries (from the energy output):")
    print(f"  {'geometry':10s} {'Geant4':>8s} {'model':>8s} {'baseline':>9s}")
    pred_yield = (pred_e.exp() - EMPTY_BIN_YIELD).clamp(min=0).cpu().numpy()
    pred_angle = (pred_a.exp() - EMPTY_BIN_YIELD).clamp(min=0).cpu().numpy()
    baseline_total = (mean_e.exp() - EMPTY_BIN_YIELD).clamp(min=0)[neutron].sum().item()
    for index, name in enumerate(test_names):
        print(f"  {name:10s} {test_energy[index, neutron].sum():8.3f} "
              f"{pred_yield[index, neutron].sum():8.3f} {baseline_total:9.3f}")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results / "initial_voxel_cnn.pt")

    # Test predictions against Geant4: solid lines are Geant4, dotted lines are the model.
    fig, axes = plt.subplots(len(test_names), 2, figsize=(11, 3.2 * len(test_names)), layout="constrained")
    axes = np.atleast_2d(axes)
    for row, name in enumerate(test_names):
        energy_ax, angle_ax = axes[row]
        draw(energy_ax, test_energy[row], ENERGY_EDGES)
        draw(angle_ax, test_angle[row], ANGLE_EDGES)
        for index, particle in enumerate(PARTICLES):
            energy_ax.stairs(pred_yield[row, index], ENERGY_EDGES, color=COLORS[particle],
                             linewidth=1.5, linestyle=(0, (1, 1)), label=f"{particle}, model")
            angle_ax.stairs(pred_angle[row, index], ANGLE_EDGES, color=COLORS[particle],
                            linewidth=1.5, linestyle=(0, (1, 1)))
        style(energy_ax, "Kinetic energy (MeV)", "Yield per bin", f"{name}: energy spectrum")
        energy_ax.set_xscale("log")
        energy_ax.set_xlim(ENERGY_EDGES[0], ENERGY_EDGES[-1])
        style(angle_ax, "θ (degrees)", "Yield per bin", f"{name}: angular distribution")
        angle_ax.set_xlim(ANGLE_EDGES[0], ANGLE_EDGES[-1])
        for ax in (energy_ax, angle_ax):
            ax.set_ylim(bottom=5e-6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [f"{label}, Geant4" if "model" not in label else label for label in labels]
    fig.legend(handles, labels, loc="outside upper left", ncols=4, frameon=False)
    output = ROOT / "plots" / "results" / "initial_test_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="#fcfcfb")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
