"""Neural networks that predict particle spectra from a 3D geometry.

Run this file to check the models: python models.py
"""
import torch
from torch import nn

from dataset import ANGLE_BINS, ENERGY_BINS, GRID_SIZE, PARTICLES


def conv_block(in_channels, out_channels):
    """Two 3D convolutions."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),  # unlike BatchNorm, works with very small batches
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(),
    )


class VoxelCNN(nn.Module):
    """3D-CNN from a voxel grid to energy spectra and angular distributions.

    Input: tensor of shape (batch, 2, 36, 36, 36), where z is the beam axis and the beam passes
    through the center of the x-y plane. Channel 0 is density in g/cm^3 and channel 1 is atomic
    number Z, both in physical units and both 0 for vacuum. Voxels must be the same physical size
    in every sample, because the network has no other way to know how big an object is.

    Output: (energy, angle), with shapes (batch, 2, ENERGY_BINS) and (batch, 2, ANGLE_BINS). The
    second dimension follows PARTICLES. Values are unbounded natural logs of the yield per
    incident proton per bin, because yields span many orders of magnitude. torch.exp() recovers
    the yields.
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        # Brings density (up to ~20) and Z (up to ~100) to similar ranges, roughly 0 to 1
        self.register_buffer("input_scale", torch.tensor([10.0, 100.0]).view(1, 2, 1, 1, 1))
        self.features = nn.Sequential(
            conv_block(2, 16),
            nn.MaxPool3d(2),  # 36 -> 18
            conv_block(16, 32),
            nn.MaxPool3d(2),  # 18 -> 9
            conv_block(32, 64),
            # 9 -> 3, so nothing is dropped and the middle cell sits on the beam axis. Flatten this
            # coarse grid instead of averaging it all away, because where the material sits
            # relative to the beam matters.
            nn.AvgPool3d(3),
            nn.Flatten(),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 3**3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.energy_head = nn.Linear(512, len(PARTICLES) * ENERGY_BINS)
        self.angle_head = nn.Linear(512, len(PARTICLES) * ANGLE_BINS)

    def forward(self, voxels):
        hidden = self.shared(self.features(voxels / self.input_scale))
        energy = self.energy_head(hidden).view(-1, len(PARTICLES), ENERGY_BINS)
        angle = self.angle_head(hidden).view(-1, len(PARTICLES), ANGLE_BINS)
        return energy, angle


if __name__ == "__main__":
    model = VoxelCNN().eval()

    # One Al cube, an Al cube followed by a lead cube, and an off-axis Al cube
    al = torch.tensor([2.699, 13.0]).view(2, 1, 1, 1)  # density in g/cm^3, Z
    pb = torch.tensor([11.35, 82.0]).view(2, 1, 1, 1)
    voxels = torch.zeros(3, 2, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    voxels[0, :, 16:20, 16:20, 16:20] = al
    voxels[1, :, 16:20, 16:20, 16:20] = al
    voxels[1, :, 16:20, 16:20, 20:24] = pb
    voxels[2, :, 0:4, 0:4, 16:20] = al

    with torch.no_grad():
        energy, angle = model(voxels)
    assert energy.shape == (3, 2, ENERGY_BINS), energy.shape
    assert angle.shape == (3, 2, ANGLE_BINS), angle.shape
    assert not torch.allclose(energy[0], energy[2]), "model can't see position relative to the beam"

    print(f"OK: VoxelCNN has {sum(p.numel() for p in model.parameters()):,} parameters")
