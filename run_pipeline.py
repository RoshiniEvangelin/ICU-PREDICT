"""Run the whole ICU-PREDICT pipeline.

    python run_pipeline.py --demo            # synthetic MIMIC-IV-format data (no credentials needed)
    python run_pipeline.py                   # real MIMIC-IV at paths.mimic_dir in config.yaml
    python run_pipeline.py --mimic-dir /data/mimiciv/3.1
    python run_pipeline.py --from-step train # rerun from a later step
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from icu_predict.config import load_config  # noqa: E402

STEPS = ["demo", "extract", "dataset", "train", "forecast", "plots"]


def main() -> None:
    ap = argparse.ArgumentParser(description="ICU-PREDICT pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--demo", action="store_true", help="use synthetic MIMIC-IV-format data")
    ap.add_argument("--mimic-dir", default=None, help="override paths.mimic_dir")
    ap.add_argument("--from-step", choices=STEPS[1:], default=None)
    args = ap.parse_args()

    overrides = {"paths": {"mimic_dir": args.mimic_dir}} if args.mimic_dir else None
    cfg = load_config(args.config, overrides)
    start = STEPS.index(args.from_step) if args.from_step else 0
    t0 = time.time()

    from icu_predict.data import extract, generate_demo
    from icu_predict.features import make_dataset
    from icu_predict.models import forecast, plots, train

    for step in STEPS[start:]:
        print(f"\n===== {step.upper()} =====")
        if step == "demo":
            if args.demo and not (Path(cfg["paths"]["demo_dir"]) / "icu/chartevents.csv.gz").exists():
                generate_demo.generate_demo(cfg["paths"]["demo_dir"], cfg["demo"]["n_patients"], cfg["demo"]["seed"])
            else:
                print("skipped" if not args.demo else "demo data already present")
        elif step == "extract":
            extract.run(cfg, use_demo=args.demo)
        elif step == "dataset":
            make_dataset.run(cfg)
        elif step == "train":
            train.run(cfg)
        elif step == "forecast":
            forecast.run(cfg)
        elif step == "plots":
            plots.run(cfg)
    import json
    src = "synthetic" if args.demo else ("mimic-iv-demo" if "demo" in Path(cfg["paths"]["mimic_dir"]).name.lower() else "mimic-iv")
    rep = Path(cfg["paths"]["outputs_dir"]) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "data_source.json").write_text(json.dumps({"source": src}), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s. Next: start the API (uvicorn app.api:app) and dashboard "
          f"(streamlit run app/dashboard.py).")


if __name__ == "__main__":
    main()
