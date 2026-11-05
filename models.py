"""Neural networks that predict particle spectra from a 3D geometry.

Run this file to check the models: python models.py
"""
import torch
from torch import nn

# Output format. The bins must match ENERGY_BINS and THETA_BINS in plot_particles.py.
PARTICLES = ("proton", "neutron")
ENERGY_BINS = 100  # 1 MeV bins, 0-100 MeV
ANGLE_BINS = 90  # 1 degree bins, 0-90 degrees from the beam axis


def conv_block(in_channels, out_channels):
    """Two 3D convolutions, then halve the grid size."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),  # unlike BatchNorm, works with very small batches
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(),
        nn.MaxPool3d(2),
    )


class VoxelCNN(nn.Module):
    """3D-CNN from a voxel grid to energy spectra and angular distributions.

    Input: tensor of shape (batch, 2, x, y, z), where z is the beam axis and the beam passes
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
            conv_block(16, 32),
            conv_block(32, 64),
            # Flatten a coarse grid instead of averaging it all away. The beam runs through one
            # fixed place, so where the material sits relative to that axis matters.
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 4**3, 512),
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

    # 32^3 grids: one Al block, an Al block followed by a lead block, and an off-axis Al block
    al = torch.tensor([2.70, 13.0]).view(2, 1, 1, 1)  # density in g/cm^3, Z
    pb = torch.tensor([11.35, 82.0]).view(2, 1, 1, 1)
    voxels = torch.zeros(3, 2, 32, 32, 32)
    voxels[0, :, 12:20, 12:20, 12:20] = al
    voxels[1, :, 12:20, 12:20, 4:8] = al
    voxels[1, :, 12:20, 12:20, 20:24] = pb
    voxels[2, :, 0:8, 0:8, 12:20] = al  # same as sample 0 but moved off the beam axis

    with torch.no_grad():
        energy, angle = model(voxels)
    assert energy.shape == (3, 2, ENERGY_BINS), energy.shape
    assert angle.shape == (3, 2, ANGLE_BINS), angle.shape
    assert not torch.allclose(energy[0], energy[2]), "model can't see position relative to the beam"

    # Other grid sizes work too
    energy, angle = model(torch.zeros(1, 2, 24, 24, 48))
    assert energy.shape == (1, 2, ENERGY_BINS) and angle.shape == (1, 2, ANGLE_BINS)

    print(f"OK: VoxelCNN has {sum(p.numel() for p in model.parameters()):,} parameters")
