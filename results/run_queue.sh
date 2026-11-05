#!/bin/bash
# Finishes the three-cube simulations, then runs both trainings. Reruns skip finished simulations.
cd "/Users/brightenjiang/Documents/fall_2026/236/final project" || exit 1
PY=.venv/bin/python
echo "start $(date)"
nix develop ~/NixForRatpac --command python3 generate_3cube.py 2>&1 | grep -v 'already done' | tail -5
train=$(ls data/3_cube/train/*.csv | wc -l); val=$(ls data/3_cube/val/*.csv | wc -l); test=$(ls data/3_cube/test/*.csv | wc -l)
leaks=$(ls data/3_cube/train/line3_* data/3_cube/val/line3_* 2>/dev/null | wc -l)
echo "train $train, val $val, test $test, line3 outside test $leaks"
if [ "$train" -ne 347 ] || [ "$val" -ne 49 ] || [ "$test" -ne 104 ] || [ "$leaks" -ne 0 ]; then echo "Dataset is not as planned, not training"; exit 1; fi
echo "3_cube training $(date)"
$PY train.py --name 3_cube --train-dir data/3_cube/train --val-dir data/3_cube/val --test-dir data/3_cube/test || exit 1
echo "off_axis training $(date)"
$PY train.py --name off_axis || exit 1
echo "FINISHED $(date)"
