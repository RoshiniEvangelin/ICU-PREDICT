#!/bin/sh
# Train on first start (MIMIC-IV if mounted at /data/mimic, otherwise synthetic demo data), then serve.
set -e
if [ ! -f outputs/models/deterioration_model.joblib ] || [ "${RETRAIN:-0}" = "1" ]; then
  if [ -f /data/mimic/icu/icustays.csv.gz ] || [ -f /data/mimic/icu/icustays.csv ]; then
    echo "[entrypoint] training on MIMIC-IV at /data/mimic"
    python run_pipeline.py --mimic-dir /data/mimic
  else
    echo "[entrypoint] no MIMIC-IV mounted at /data/mimic — training on synthetic demo data"
    python run_pipeline.py --demo
  fi
fi
exec python -m uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-8000}"
