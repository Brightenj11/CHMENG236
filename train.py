#!/usr/bin/env python3
"""Train VoxelCNN on one dataset and evaluate it on geometries it has never seen.

Usage:
    python train.py
    python train.py --name 3_cube --train-dir data/3_cube/train --val-dir data/3_cube/val \\
        --test-dir data/3_cube/test
"""
import argparse
import copy
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
from symmetry import augment, canonicalize

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


def test_group(name):
    """Which kind of unseen geometry this is, so one average can't hide a weak spot."""
    if name.startswith("line3"):
        return "3 cubes in a line"
    if name.startswith("offset_corner"):
        return "unseen corner offset"
    if name.startswith("both_staircase"):
        return "unseen staircase offset"
    return "2-cube lines"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="initial", help="prefix for saved models and plots")
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data" / "initial_train")
    parser.add_argument("--val-dir", type=Path, default=ROOT / "data" / "initial_val")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "initial_test")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    # CPU by default: on an M1, the macOS GPU watchdog killed training on "mps" with a batch of 16.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--angle-weight", type=float, default=1.0,
                        help="loss = energy error + angle-weight * angle error")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--symmetry", choices=("none", "augment", "canonical"), default="none",
                        help="augment: random rotations and z shifts in training. "
                             "canonical: move every geometry to one standard position (exact)")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many epochs without a better val loss, and keep the "
                             "best model (0: train all epochs)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    train_names, train_voxels, train_energy, train_angle = load_dataset(args.train_dir)
    val_names, val_voxels, val_energy, val_angle = load_dataset(args.val_dir)
    test_names, test_voxels, test_energy, test_angle = load_dataset(args.test_dir)
    shared = (set(train_names) & set(val_names)) | (set(train_names) & set(test_names)) | (set(val_names) & set(test_names))
    if shared:
        raise SystemExit(f"Same geometry in more than one dataset: {shared}")
    print(f"{len(train_names)} train, {len(val_names)} val and {len(test_names)} test geometries on {device}")

    train_x = torch.as_tensor(train_voxels, device=device)
    val_x = torch.as_tensor(val_voxels, device=device)
    test_x = torch.as_tensor(test_voxels, device=device)
    if args.symmetry == "canonical":
        train_x, val_x, test_x = canonicalize(train_x), canonicalize(val_x), canonicalize(test_x)
    train_e, train_a = to_log(train_energy).to(device), to_log(train_angle).to(device)
    val_e, val_a = to_log(val_energy).to(device), to_log(val_angle).to(device)
    test_e, test_a = to_log(test_energy).to(device), to_log(test_angle).to(device)

    def combined_loss(pred_e, pred_a, true_e, true_a):
        return nn.functional.mse_loss(pred_e, true_e) + args.angle_weight * nn.functional.mse_loss(pred_a, true_a)

    # Baseline: always predict the average training spectrum. Geometries the beam misses are left
    # out: they have no neutrons, so they would drag the average down and flatter the model.
    hit = torch.tensor([not name.startswith("missed") for name in train_names], device=device)
    mean_e, mean_a = train_e[hit].mean(dim=0), train_a[hit].mean(dim=0)
    baseline_val_loss = combined_loss(mean_e.expand_as(val_e), mean_a.expand_as(val_a), val_e, val_a).item()
    print(f"baseline val loss {baseline_val_loss:.4f}")

    model = VoxelCNN().to(device)
    # Start from the baseline, so the network only has to learn how each geometry differs from it.
    with torch.no_grad():
        model.energy_head.bias.copy_(mean_e.flatten())
        model.angle_head.bias.copy_(mean_a.flatten())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in torch.randperm(len(train_x), device=device).split(args.batch_size):
            inputs = augment(train_x[batch]) if args.symmetry == "augment" else train_x[batch]
            pred_e, pred_a = model(inputs)
            loss = combined_loss(pred_e, pred_a, train_e[batch], train_a[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = combined_loss(*model(val_x), val_e, val_a).item()
        history.append((epoch, float(np.mean(losses)), val_loss))
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:4d}   train loss {history[-1][1]:.4f}   val loss {val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif args.patience and epoch - best_epoch >= args.patience:
            break

    if args.patience:
        print(f"Stopped after epoch {len(history)}. Keeping epoch {best_epoch}, val loss {best_val:.4f}")
        model.load_state_dict(best_state)

    # The test set is only evaluated here, once, after training is finished.
    with torch.no_grad():
        final_e, final_a = model(train_x)
        final_val_e, final_val_a = model(val_x)
        pred_e, pred_a = model(test_x)

    print("\nLog-space RMSE per bin (lower is better, 0.69 is about a factor of 2):")
    report("train", final_e, final_a, train_e, train_a)
    report("val", final_val_e, final_val_a, val_e, val_a)
    report("test", pred_e, pred_a, test_e, test_a)
    report("baseline", mean_e.expand_as(test_e), mean_a.expand_as(test_a), test_e, test_a)

    neutron = PARTICLES.index("neutron")
    pred_yield = (pred_e.exp() - EMPTY_BIN_YIELD).clamp(min=0).cpu().numpy()
    pred_angle = (pred_a.exp() - EMPTY_BIN_YIELD).clamp(min=0).cpu().numpy()
    baseline_total = (mean_e.exp() - EMPTY_BIN_YIELD).clamp(min=0)[neutron].sum().item()
    groups = np.array([test_group(name) for name in test_names])
    true_total, pred_total = test_energy[:, neutron].sum(axis=1), pred_yield[:, neutron].sum(axis=1)

    print("\nTest results by group:")
    for group in sorted(set(groups)):
        members = torch.as_tensor(groups == group, device=device)
        chosen = groups == group
        print(f"{group} ({chosen.sum()} geometries)")
        report("model", pred_e[members], pred_a[members], test_e[members], test_a[members])
        report("baseline", mean_e.expand_as(test_e[members]), mean_a.expand_as(test_a[members]),
               test_e[members], test_a[members])
        model_error = np.abs(pred_total[chosen] - true_total[chosen]) / true_total[chosen]
        baseline_error = np.abs(baseline_total - true_total[chosen]) / true_total[chosen]
        print(f"  mean error in total neutrons: model {model_error.mean():.1%}   baseline {baseline_error.mean():.1%}")

    print("\nNeutrons per incident proton, test geometries (from the energy output):")
    print(f"  {'geometry':30s} {'Geant4':>8s} {'model':>8s} {'baseline':>9s}")
    for index, name in enumerate(test_names):
        print(f"  {name:30s} {true_total[index]:8.3f} {pred_total[index]:8.3f} {baseline_total:9.3f}")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results / f"{args.name}_voxel_cnn.pt")
    np.savetxt(results / f"{args.name}_history.csv", history, delimiter=",", comments="",
               header="epoch,train_loss,val_loss", fmt=["%d", "%.6f", "%.6f"])

    # Loss over time. Train loss is the running average with dropout on, and val loss has it off.
    epochs, train_losses, val_losses = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(epochs, train_losses, color=COLORS["proton"], linewidth=1.5, label="Train")
    ax.plot(epochs, val_losses, color=COLORS["neutron"], linewidth=1.5, label="Validation")
    ax.axhline(baseline_val_loss, color="#52514e", linewidth=1, linestyle=(0, (4, 3)),
               label="Baseline on validation")
    style(ax, "Epoch", "Loss (energy + angle)", "Loss over training")
    ax.set_xlim(1, len(history))
    ax.legend(frameon=False)
    curve = ROOT / "plots" / "results" / f"{args.name}_training_curve.png"
    curve.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve, dpi=150, facecolor="#fcfcfb")
    print(f"\nSaved {curve}")

    # Test predictions against Geant4: solid lines are Geant4, dotted lines are the model.
    # Up to 4 evenly spaced geometries from each group, since a plot of every one would be unreadable.
    shown = []
    for group in sorted(set(groups)):
        members = np.flatnonzero(groups == group)
        shown += [members[i] for i in np.unique(np.linspace(0, len(members) - 1, 4).astype(int))]
    fig, axes = plt.subplots(len(shown), 2, figsize=(11, 3.2 * len(shown)), layout="constrained")
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(shown):
        name = test_names[sample]
        energy_ax, angle_ax = axes[row]
        draw(energy_ax, test_energy[sample], ENERGY_EDGES)
        draw(angle_ax, test_angle[sample], ANGLE_EDGES)
        for index, particle in enumerate(PARTICLES):
            energy_ax.stairs(pred_yield[sample, index], ENERGY_EDGES, color=COLORS[particle],
                             linewidth=1.5, linestyle=(0, (1, 1)), label=f"{particle}, model")
            angle_ax.stairs(pred_angle[sample, index], ANGLE_EDGES, color=COLORS[particle],
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
    output = ROOT / "plots" / "results" / f"{args.name}_test_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="#fcfcfb")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
