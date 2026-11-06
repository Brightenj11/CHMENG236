"""Turn Geant4 runs in data/<dataset>/ into model inputs and targets.

Every run is a pair of files: <name>.mac holds the cube list and <name>_nt_hits_t0.csv holds the
detected particles.

Run this file to check the geometry translation: python dataset.py
"""
import csv
from pathlib import Path

import numpy as np

PARTICLES = ("proton", "neutron")

# Log-spaced energy bins, because most neutrons are low energy. Below 0.1 MeV is not counted, which
# drops about 0.25% of neutrons. Nothing in the data is above 100 MeV.
ENERGY_EDGES = np.logspace(-1, 2, 101)  # MeV
ANGLE_EDGES = np.linspace(0, 180, 181)  # degrees from the beam axis
ENERGY_BINS = len(ENERGY_EDGES) - 1
ANGLE_BINS = len(ANGLE_EDGES) - 1

# Added to every bin before taking the log, so empty bins are defined. It must stay well below the
# value of one count, which is 1 / primaries.
EMPTY_BIN_YIELD = 1e-6

# Must match the lattice in src/DetectorConstruction.cc. Position (4, 4, *) is on the beam axis.
LATTICE_SIZE = 9
VOXELS_PER_CUBE = 4  # 2.5 mm voxels for 1 cm cubes
GRID_SIZE = LATTICE_SIZE * VOXELS_PER_CUBE

# NIST material: (density in g/cm^3, atomic number Z), using the values Geant4 uses
MATERIALS = {
    "G4_Al": (2.699, 13),
    "G4_Cu": (8.96, 29),
    "G4_W": (19.3, 74),
    "G4_Pb": (11.35, 82),
    "G4_Ag": (10.5, 47),  # never in training: the unseen-material test
}


def load_hits(path):
    """Return {particle: (energies in MeV, thetas in degrees)}. Lines starting with # are header."""
    columns = {}
    with open(path, newline="") as f:
        for _event, particle, energy, theta, _phi in csv.reader(
            line for line in f if not line.startswith("#")
        ):
            energies, thetas = columns.setdefault(particle, ([], []))
            energies.append(float(energy))
            thetas.append(float(theta))
    return {name: (np.array(e), np.array(t)) for name, (e, t) in columns.items()}


def read_macro(path):
    """Return (cubes, primaries) from a run macro. Each cube is (i, j, k, material)."""
    cubes, primaries = [], None
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["/geom/addCube"]:
            cubes.append((int(fields[1]), int(fields[2]), int(fields[3]), fields[4]))
        elif fields[:1] == ["/run/beamOn"]:
            primaries = int(fields[1])
    if primaries is None:
        raise ValueError(f"No /run/beamOn in {path}")
    return cubes, primaries


def voxelize(cubes):
    """Return a (2, 36, 36, 36) grid. Channel 0 is density, channel 1 is Z, and vacuum is 0."""
    grid = np.zeros((2, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for i, j, k, material in cubes:
        if not all(0 <= index < LATTICE_SIZE for index in (i, j, k)):
            raise ValueError(f"Cube ({i}, {j}, {k}) is outside the lattice")
        x, y, z = (index * VOXELS_PER_CUBE for index in (i, j, k))
        grid[:, x:x + VOXELS_PER_CUBE, y:y + VOXELS_PER_CUBE, z:z + VOXELS_PER_CUBE] = np.reshape(
            MATERIALS[material], (2, 1, 1, 1)
        )
    return grid


def transform_cubes(cubes, quarter_turns=0, mirror=False, z_shift=0):
    """Move a cube list by a symmetry of the setup: mirror x, then turn around the beam, then shift
    along it. Matches symmetry.rotate on voxel grids, which symmetry.py checks."""
    last = LATTICE_SIZE - 1
    moved = []
    for i, j, k, material in cubes:
        if mirror:
            i = last - i
        for _ in range(quarter_turns):
            i, j = last - j, i
        moved.append((i, j, k + z_shift, material))
    return moved


def spectra(hits, primaries):
    """Return energy (2, ENERGY_BINS) and angle (2, ANGLE_BINS) yields per incident proton per bin.

    Normalizes by protons fired, because events with no detected particle leave no row.
    """
    empty = (np.array([]), np.array([]))
    energy = [np.histogram(hits.get(p, empty)[0], ENERGY_EDGES)[0] for p in PARTICLES]
    angle = [np.histogram(hits.get(p, empty)[1], ANGLE_EDGES)[0] for p in PARTICLES]
    return np.array(energy, dtype=np.float32) / primaries, np.array(angle, dtype=np.float32) / primaries


def load_dataset(directory):
    """Return (names, voxels, energy, angle) as arrays with one row per geometry."""
    names, voxels, energies, angles = [], [], [], []
    for macro in sorted(Path(directory).glob("*.mac")):
        cubes, primaries = read_macro(macro)
        energy, angle = spectra(load_hits(macro.with_name(f"{macro.stem}_nt_hits_t0.csv")), primaries)
        names.append(macro.stem)
        voxels.append(voxelize(cubes))
        energies.append(energy)
        angles.append(angle)
    if not names:
        raise ValueError(f"No runs found in {directory}")
    return names, np.stack(voxels), np.stack(energies), np.stack(angles)


if __name__ == "__main__":
    # Al then Pb along the beam: each cube fills exactly its own 4 x 4 x 4 voxels.
    grid = voxelize([(4, 4, 4, "G4_Al"), (4, 4, 5, "G4_Pb")])
    assert grid.shape == (2, 36, 36, 36)
    assert (grid[0] > 0).sum() == 2 * 4**3
    assert np.allclose(grid[:, 16:20, 16:20, 16:20], np.reshape((2.699, 13), (2, 1, 1, 1)))
    assert np.allclose(grid[:, 16:20, 16:20, 20:24], np.reshape((11.35, 82), (2, 1, 1, 1)))
    # The lattice is centered on the beam: mirroring x and y leaves this on-axis shape unchanged.
    assert np.array_equal(grid, grid[:, ::-1, ::-1, :])

    hits = {"neutron": (np.array([1.0, 1.0, 50.0]), np.array([10.5, 120.5, 179.5]))}
    energy, angle = spectra(hits, primaries=10)
    assert energy[0].sum() == 0 and np.isclose(energy[1].sum(), 0.3) and np.isclose(angle[1].sum(), 0.3)
    assert np.isclose(angle[1][120], 0.1)
    print("OK: dataset checks passed")
