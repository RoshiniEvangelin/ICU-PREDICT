"""Join features, deterioration labels and forecast targets into one modelling table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icu_predict.config import forecast_horizons, load_config
from icu_predict.features.build import build_features
from icu_predict.features.labels import build_forecast_targets, build_labels


def run(cfg: dict) -> pd.DataFrame:
    proc = Path(cfg["paths"]["processed_dir"])
    cohort = pd.read_parquet(proc / "cohort.parquet")
    timeline = pd.read_parquet(proc / "timeline.parquet")
    feats = build_features(timeline, cohort, cfg)
    labels = build_labels(timeline, cohort, cfg)
    targets = build_forecast_targets(timeline, cfg["forecast"]["targets"], forecast_horizons(cfg))
    ds = (feats.merge(labels[["stay_id", "hour", "label", "eligible", "hours_to_event", "unstable"]], on=["stay_id", "hour"])
               .merge(targets, on=["stay_id", "hour"])
               .merge(cohort[["stay_id", "subject_id"]], on="stay_id"))
    ds.to_parquet(proc / "dataset.parquet", index=False)
    labels.to_parquet(proc / "labels.parquet", index=False)
    el = ds[ds["eligible"] == 1]
    print(f"[dataset] {len(ds):,} stay-hours; {len(el):,} eligible prediction points across "
          f"{el['stay_id'].nunique():,} stays; positive rate {el['label'].mean():.2%}")
    ev = labels.groupby("stay_id")[["ev_instability", "ev_vasopressor", "ev_death"]].max().mean()
    print(f"[dataset] stays with each event type:\n{ev.round(3).to_string()}")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
