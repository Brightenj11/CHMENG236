#!/bin/bash
# Symmetry comparison: Geant4 check, 3 seeds each of augmentation and canonicalization, then evaluation.
cd "/Users/brightenjiang/Documents/fall_2026/236/final project" || exit 1
PY=.venv/bin/python
echo "start $(date)"
nix develop ~/NixForRatpac --command python3 generate_symmetry_check.py 2>&1 | tail -9
for seed in 0 1 2; do
  for method in canonical augment; do
    echo "sym_${method}_s${seed} $(date)"
    $PY train.py --name sym_${method}_s${seed} --symmetry $method --seed $seed --patience 30 \
      --train-dir data/3_cube/train --val-dir data/3_cube/val --test-dir data/3_cube/test 2>&1 \
      | grep -E 'geometries|^epoch|Stopped' || exit 1
  done
done
echo "evaluation $(date)"
$PY symmetry_eval.py > results/symmetry_comparison.txt 2>&1 || exit 1
rm -rf __pycache__
echo "FINISHED $(date)"
