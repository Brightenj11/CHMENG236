# CLAUDE.md

CHMENG 236 final project: **A Comparative Study of 3D Input Architectures for Neural Network
Predictions of MeV Proton Scattering** (see `Project Proposal.pdf`). Geant4 simulates 100 MeV protons
on 3D geometries made of 1 cm cubes. Neural networks take the geometry as input and predict the
proton and neutron energy spectra and angular distributions. The proposal compares voxel grids
(3D-CNN), point clouds (PointNet) and graphs (GNN). **Only the 3D-CNN exists so far.**

## Working rules
- **Running Geant4:** it runs only inside Nix: `nix develop ~/NixForRatpac` (the user's alias is `geant4dev`). Build with `cmake -B build && cmake --build build`, which gives `build/exampleB1`. If CMake can't find Geant4, you are outside Nix.
- **Running Python:** use the project venv, `.venv/bin/python`. The data-generation scripts call Geant4, so they run under Nix with `python3`.
- **Device:** train on the CPU. On this M1, the MPS backend crashed (GPU watchdog, `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`).
- **Long runs:** start them detached with `nohup caffeinate -is <script> > results/<x>.log 2>&1 &`, so they survive the Claude session ending. Jobs tied to a session were killed once. Closing the lid still pauses runs; they continue on wake.
- **Git:** don't commit unless the user asks.
- **Checking runs:** read the log before saying a run has finished. Checkpoint names have been mixed up before.
- **Protons and neutrons matter equally.** Report both. Proton errors are larger (about 1.1 in log space for every model) because most protons stop in the cubes, which leaves their spectra sparse. Proton results still count as much as neutron results.

## Simulation (Geant4, `src/`, `include/`)
- **Beam:** a proton gun, 100 MeV, at z = -50 cm, pointing along +z. Fixed.
- **Lattice:** a 9×9×9 lattice of possible 1 cm cube positions, centered at the origin.
  - Position `(i,j,k)` is placed at `((i-4), (j-4), (k-4))` cm.
  - `(4,4,*)` is on the beam axis, and the lattice's front face is at z = -4.5 cm.
  - The macro command is `/geom/addCube i j k G4_Al`. It is a custom `G4UIcmdWithAString`, because `G4GenericMessenger` keeps only the first word of a string argument.
- **Materials:** G4_Al, G4_Cu, G4_W, G4_Pb, in a G4_Galactic world. Physics list QBBC.
- **Detector:** a full sphere shell, radius 1.3 m and 1 mm thick, around everything.
  - `SteppingAction` kills every track entering it and records protons and neutrons: energy, theta and phi.
  - It started as a plane, then a hemisphere. It became a full sphere because about 46% of neutrons go backward.
  - The gun is inside the sphere, so the incoming beam is never counted.
- **Output:** one CSV per run, `<name>_nt_hits_t0.csv`, with columns event, particle, energy_MeV, theta_deg, phi_deg.
- **Seeds:** each run's seed comes from the crc32 of the geometry name, so reruns are byte-identical.
- **Sanity checks passed:**

  | Geometry | Result |
  |---|---|
  | 1 cm Al | proton peak at about 83–84 MeV |
  | 1 cm Cu | about 45 MeV |
  | 1 cm Pb | about 51 MeV |
  | 1 cm W | stops nearly all protons |
  | 2 Al cubes | about 64 MeV |
  | Beam misses the cubes | exactly 100k protons at 100 MeV and no neutrons |

## Python pipeline
| File | Purpose |
|---|---|
| `dataset.py` | Reads `.mac` and CSV files, builds the voxel grid and the target spectra. `transform_cubes` moves cube lists by the symmetries. |
| `models.py` | `VoxelCNN` |
| `train.py` | Training, evaluation and plots. Options: `--name --train-dir --val-dir --test-dir --epochs 300 --batch-size 4 --lr 1e-3 --angle-weight 1 --seed --symmetry {none,augment,canonical} --patience` |
| `symmetry.py` | Rotations and mirrors, z shifts, `augment()` and `canonicalize()` |
| `symmetry_eval.py` | The symmetry comparison table, plus the Geant4 symmetry check |
| `generate_data.py` | Builds `data/initial_*` (Nix) |
| `generate_3cube.py` | Builds `data/3_cube/*` (Nix) |
| `generate_symmetry_check.py` | Builds `data/symmetry_check` (Nix) |
| `plot_particles.py` | Spectrum and angle plot for a single run, written to `plots/<folder>/` |

Each module checks itself when run directly (`python dataset.py`, `symmetry.py`, `models.py`).

- **Input:** a 36³ voxel grid at 2.5 mm, so each cube is 4×4×4 voxels.
  - Two channels: density in g/cm³ and atomic number Z. Empty space is 0.
  - Voxels smaller than a cube leave room for half blocks or curved shapes later.
- **Targets:**
  - Energy: 100 log-spaced bins from 0.1 to 100 MeV, chosen because most neutrons are low energy.
  - Angle: 180 one-degree theta bins.
  - Both are per particle, as yield per incident proton.
  - The model predicts log(yield + 1e-6), so empty bins are defined.
- **VoxelCNN, 1.39M parameters:**
  - 3 conv blocks (16, 32, 64 channels) with GroupNorm.
  - MaxPool takes the grid from 36 to 18 to 9.
  - AvgPool3d(3) takes it from 9 to 3. With an odd size, the center cell sits on the beam; an earlier 9→4 pool cut off a slice.
  - Then a linear layer to 512, and separate energy and angle heads.
  - The head biases start at the mean training spectrum.
- **Training:** Adam with a constant lr of 1e-3; the user wants it kept constant for now and the effort put into data. Batch size 4.
  - Loss is MSE(energy) + angle_weight · MSE(angle), in log space.
  - Runs are seeded and deterministic.
- **Metric:** log-space RMSE per bin, reported for each test group, plus the error in total neutron count.

## Story so far

### 1. First model: `initial` run (44 geometries)
- **Data:** single cubes and 2-cube lines along the beam. The model is tested on 2-cube lines with **material pairs it never saw**: (Al,Pb), (Cu,Al), (Pb,Cu) and (Cu,W).
- **Why:** this is the smallest test of whether the model learns more than memorizing its training geometries.
- **Result:** neutron RMSE 0.361 for energy and 0.362 for angle.
- **Caveat:** `data/initial_*` was later grown in place, so this run can't be repeated exactly.

### 2. More geometries: `off_axis` run (`data/initial_*`: 172 train, 20 val, 4 test)
- **Data added:**
  - Off-axis layouts: diagonal, beside, two behind.
  - 16 layouts where the beam misses one or both cubes.
- **Why:** more variety in geometry, at the user's request.
- **Result:**
  - The 4 test pairs: RMSE 0.342, total neutron error 16.4%.
  - Best validation at epoch 84.
- **A wrong guess, since disproved:** I first blamed the beam-miss cases for Al-first lines being underpredicted. The 3_cube comparison has the same 60 misses, so that wasn't the cause.

### 3. Three cubes: `3_cube` run (`data/3_cube`: 347 train, 49 val, 104 test)
- **The user's goal:** train on fewer cubes, then **test on 3-cube lines**, to see whether the model really learns the physics.
  - All 64 3-cube lines are test-only.
  - Two offset layouts are also test-only: "corner" (12) and "staircase" (24).
  - Training gets 3-cube shapes that are not lines: one behind with one offset, offset by different amounts, both offset.
- **Result:**

  | Test group | Neutron energy RMSE | Total neutron error |
  |---|---|---|
  | 3-cube lines | 0.369 | 17.2% (the mean-spectrum baseline gives 35.7%) |
  | Corner | — | 10.5% |
  | Staircase | — | 5.8% |
  | 2-cube pairs | 0.466 | 27.9% |

- **Weak spots:**
  - Al-first lines are underpredicted (28.6%).
  - Al on the axis with heavy cubes off it is underpredicted.
  - Validation plateaus around epoch 20.
- **Conclusion:** more 3-cube data helped the offset layouts but not the lines. The model gets the **shape** of the spectra right; the remaining error is mostly in the **overall scale**.
- **Deferred:** the user wants a metric separating shape error from scale error, "maybe later".

### 4. Symmetries: `sym_*` runs (`results/symmetry_queue.sh`, finished Nov 5 2026)
- **Exact symmetries of the setup:**
  - Rotations and mirrors around the beam axis. There are 8 (the D4 group), because theta and energy don't depend on phi.
  - Shifts along z by whole cubes, because the world is vacuum and the detector records direction, not position.
- **Question:** is it better to build the symmetries in or let the model learn them? The current `3_cube` model is the reference, not a fair competitor.
- **Methods compared:**
  - **Augmentation:** a random rotation, mirror and z shift for each sample.
  - **Canonicalization:** first shift the front cube to slice 0, then pick the rotation or mirror that puts the most off-axis mass toward +x, then toward +y. This is frame averaging with a frame of a single element, and it is exactly invariant.
  - Each was trained with 3 seeds and `--patience 30`.
- **Tests:**
  - **Original test set:** ordinary accuracy.
  - **Moved copies:** each test geometry in up to 24 versions (8 rotations × front, original and back positions). The correct answers are unchanged, so no new simulations are needed.
  - **Consistency:** the spread of one geometry's predictions across its copies. This was the user's idea; 0 means exactly symmetric.
  - **Geant4 check:** 3 test geometries simulated rotated, shifted and with a new seed. This shows the symmetries hold in the physics, beyond noise.
- **Result:** neutron energy, all 104 test geometries; full table in `results/symmetry_comparison.txt`.

  | Model | Original | Moved | Consistency |
  |---|---|---|---|
  | No symmetry (`3_cube`, 1 seed) | **0.365** | 3.114 | 2.044 |
  | Augmentation (3 seeds) | 0.443 ± 0.077 | 0.452 | 0.126 |
  | Canonicalization (3 seeds) | 0.438 ± 0.005 | 0.438 | **0.000** |

- **What the result means:**
  - The plain model has **not** learned the symmetries; its error is about 8× worse on moved copies. Even the on-axis lines get worse, from z shifts alone.
  - Augmentation and canonicalization tie on accuracy. Canonicalization is exact, much steadier across seeds, and about 3× faster to train.
  - Building in symmetry costs about 0.07 on the original positions. Possibly the canonical rule sends nearly identical shapes to different positions. Not tested.
  - The Geant4 check passed: moved geometries differ from the original about as much as a reseeded run does, and total neutrons agree within about 3%. The one exception, staircase shifted at +6.5%, is probably noise.
- **Caveats:**
  - Augmentation seed 2 stalled at validation loss 2.43, against about 1.74 for the others.
  - Canonicalization seed 2 stopped at epoch 23.
  - Augmentation seed 1 was still improving at epoch 300. Patience 30 is too short for augmentation.

### 5. Unseen material: single silver cube (`unseen_material.py`, `data/unseen_material/`)
- **Question:** can the model predict a material it never trained on? The test is one silver cube (G4_Ag, Z 47, 10.5 g/cm³) on the beam axis. Silver sits in the gap between Cu (29) and W (74).
- **Baselines:**
  - Copy the Cu spectrum.
  - Copy the W spectrum.
  - Interpolate the log spectra between Cu and W, weighted by Z.
- **Result:**
  - **Neutrons: good.**
    - The `3_cube` model scores 0.35 on energy and 0.34 on angle, about the same as Z interpolation (0.35 and 0.26).
    - The augmentation models do best, especially seed 1 (0.22 and 0.14, total -5%).
    - The models generally predict too few neutrons in total, between -5% and -29%.
  - **Protons: the energy peak is in the wrong place.**
    - Geant4 puts the exit peak at about 42 MeV, below Cu's 45 MeV, because a silver cube stops more protons.
    - Every model puts the peak at about 48 MeV, between Cu and Pb. It seems to interpolate by Z order rather than by stopping power.
    - The peak is only 1–2 bins wide, so the total from the energy output comes out 77–96% too low. The angle output gets the proton shape and total roughly right, so the two output heads disagree.
  - **Takeaway:** with only 4 training materials, the model can't learn energy loss as a function of density and Z. More materials would be needed.
- **Plot:** `plots/results/unseen_material_ag.png`.

## Next steps (planned, ask before starting)
- **Frame averaging, overnight:** average the prediction over all 8 rotations and mirrors, with z canonicalized. It is exactly invariant like canonicalization, but without picking a single position, so it may recover the accuracy canonicalization loses. Costs 8× the forward passes.
- **Fairer seeds:** rerun augmentation with longer patience (about 100) and more epochs, and run more seeds or report medians.
- **Other work:**
  - The shape-vs-scale metric.
  - The Al-first underprediction.
  - 3-cube lines in training.
  - Half blocks and curved shapes.
  - PointNet and GNN models (the point of the proposal).

## Outputs
- **Checkpoints:** `results/<name>_voxel_cnn.pt`.
- **Training histories:** `results/<name>_history.csv`.
- **Queue logs:** `results/*_queue.log`.
- **Plots:**
  - `plots/results/<name>_training_curve.png` and `<name>_test_predictions.png`.
  - Spectra of single runs go under `plots/<dataset>/`.
