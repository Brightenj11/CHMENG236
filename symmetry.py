"""Symmetries of the setup, for voxel grids of shape (batch, 2, x, y, z) with z along the beam.

Both are exact for this setup:
- Turning or mirroring around the beam: the outputs (energy, theta) don't depend on phi, so all
  8 rotations and mirrors of a geometry give the same spectra.
- Shifting along the beam: the space is vacuum and the detector records direction, not position.

Two ways to use them:
- augment(): random rotation, mirror and z shift each time a geometry is used in training. The
  model learns the symmetry, approximately.
- canonicalize(): move every geometry to one standard position before the model sees it. The model
  is then exactly symmetric. This is frame averaging with a frame of one element.

Run this file to check them: python symmetry.py
"""
import torch

from dataset import VOXELS_PER_CUBE

# The 8 rotations and mirrors around the beam: (quarter turns, mirror x first)
ROTATIONS = [(turns, mirror) for mirror in (False, True) for turns in range(4)]


def rotate(voxels, quarter_turns, mirror):
    """Turn (batch, 2, x, y, z) voxels around the beam axis, which runs through the grid's center."""
    if mirror:
        voxels = torch.flip(voxels, dims=(2,))
    return torch.rot90(voxels, quarter_turns, dims=(2, 3))


def z_range(sample):
    """First and last occupied z voxel of one (2, x, y, z) sample, or None if it is empty."""
    occupied = torch.nonzero((sample[0] > 0).any(dim=0).any(dim=0)).flatten()
    return None if len(occupied) == 0 else (occupied[0].item(), occupied[-1].item())


def z_shift_range(sample):
    """Smallest and largest shift along the beam, in whole cubes, that keeps the shape on the grid."""
    occupied = z_range(sample)
    if occupied is None:
        return 0, 0
    first, last = occupied
    return -(first // VOXELS_PER_CUBE), (sample.shape[-1] - 1 - last) // VOXELS_PER_CUBE


def shift_z(sample, cubes):
    """Shift one sample along the beam. Wrapping is safe, because only empty voxels wrap around."""
    return torch.roll(sample, cubes * VOXELS_PER_CUBE, dims=-1)


def augment(voxels):
    """Random rotation or mirror and a random shift along the beam, for each sample in a batch."""
    out = []
    for sample in voxels:
        turns, mirror = ROTATIONS[int(torch.randint(len(ROTATIONS), (1,)))]
        sample = rotate(sample[None], turns, mirror)[0]
        low, high = z_shift_range(sample)
        out.append(shift_z(sample, int(torch.randint(low, high + 1, (1,)))))
    return torch.stack(out)


def canonicalize(voxels):
    """Move each sample to a standard position: frontmost cube in the first lattice slice, then the
    rotation or mirror that puts the most off-axis mass toward +x, then toward +y.

    Every rotated, mirrored or shifted copy of a geometry ends up as the same grid, because the
    choice is made over all 8 versions with a rule that depends only on the grid. Exact ties, as in
    symmetric shapes, are broken by comparing the raw grids.
    """
    coords = torch.arange(voxels.shape[2], dtype=torch.float64) - (voxels.shape[2] - 1) / 2
    out = []
    for sample in voxels:
        sample = shift_z(sample, z_shift_range(sample)[0])
        best_key, best = None, None
        for turns, mirror in ROTATIONS:
            candidate = rotate(sample[None], turns, mirror)[0]
            density = candidate[0].double()
            key = (round((density.sum(dim=(1, 2)) * coords).sum().item(), 6),
                   round((density.sum(dim=(0, 2)) * coords).sum().item(), 6),
                   candidate.contiguous().numpy().tobytes())
            if best_key is None or key > best_key:
                best_key, best = key, candidate
        out.append(best)
    return torch.stack(out)


if __name__ == "__main__":
    from dataset import transform_cubes, voxelize

    cubes = [(4, 4, 4, "G4_Al"), (5, 4, 5, "G4_Pb"), (6, 4, 6, "G4_Cu"), (4, 3, 5, "G4_W")]
    original = torch.as_tensor(voxelize(cubes))[None]
    canonical = canonicalize(original)
    for turns, mirror in ROTATIONS:
        for z in (-2, 0, 2):
            # Moving the cube list and moving the voxel grid agree, so Geant4 checks use the same moves.
            moved = torch.as_tensor(voxelize(transform_cubes(cubes, turns, mirror, z)))[None]
            assert torch.equal(moved, shift_z(rotate(original, turns, mirror)[0], z)[None]), (turns, mirror, z)
            # Every copy ends up at the same standard position.
            assert torch.equal(canonicalize(moved), canonical), (turns, mirror, z)

    torch.manual_seed(0)
    batch = original.repeat(16, 1, 1, 1, 1)
    augmented = augment(batch)
    assert torch.equal(augmented.sum(dim=(1, 2, 3, 4)), batch.sum(dim=(1, 2, 3, 4)))  # nothing lost
    assert len({a.numpy().tobytes() for a in augmented}) > 1  # copies really differ
    assert all(torch.equal(canonicalize(a[None]), canonical) for a in augmented)
    print("OK: symmetry checks passed")
