"""Entry point for a Hugging Face Space using the Gradio SDK (works where Docker Spaces aren't available).

On first start it downloads the open MIMIC-IV demo (synthetic data if that fails), trains the models,
then serves the hospital UI on port 7860. Later restarts reuse the trained models.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("WARM_START", "1")

if not (ROOT / "outputs/models/deterioration_model.joblib").exists():
    print("[space] first start: training models (a few minutes)…", flush=True)
    got_data = subprocess.run([sys.executable, "scripts/fetch_mimic_demo.py", "data/raw/mimic-iv-demo"]).returncode == 0
    cmd = ["--mimic-dir", "data/raw/mimic-iv-demo"] if got_data else ["--demo"]
    subprocess.run([sys.executable, "run_pipeline.py", *cmd], check=True)

sys.path.insert(0, str(ROOT))
import uvicorn  # noqa: E402

uvicorn.run("app.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
