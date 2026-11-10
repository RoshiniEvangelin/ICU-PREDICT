#!/bin/sh
# Hugging Face Space build: train on the real (open) MIMIC-IV demo; fall back to synthetic data.
set -e
if python scripts/fetch_mimic_demo.py data/raw/mimic-iv-demo; then
  python run_pipeline.py --mimic-dir data/raw/mimic-iv-demo
else
  echo "[build] MIMIC-IV demo unavailable — training on synthetic data instead"
  python run_pipeline.py --demo
fi
