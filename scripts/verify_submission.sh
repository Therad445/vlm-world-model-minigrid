#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "README.md"
  "configs/minigrid_empty.yaml"
  "run_pipeline.py"
  "report/report.pdf"
  "outputs/metrics.csv"
  "outputs/episodes.csv"
  "outputs/training_loss.csv"
  "outputs/plots/success_rate.png"
  "outputs/plots/training_loss.png"
  "outputs/screenshots/reconstruction.png"
  "outputs/screenshots/imagined_rollout.png"
  "outputs/gifs/wm_vlm_episode.gif"
  "logs/full_run.log"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing: $path" >&2
    exit 1
  fi
done

python -m compileall src run_pipeline.py >/tmp/vlm_world_model_compile.log
cat outputs/metrics.csv

echo "submission files look complete"
