"""Download the open-access MIMIC-IV Clinical Database Demo v2.2 (100 patients, ODbL) files this project uses.

    python scripts/fetch_mimic_demo.py [out_dir]

No credentials needed. Exits non-zero if any file could not be fetched, so callers can fall back to
synthetic data. Set MIMIC_DEMO_LOCAL=/path/to/demo to copy from a local folder instead (offline builds).
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/mimic-iv-demo/2.2/"
FILES = ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/labevents.csv.gz",
         "icu/icustays.csv.gz", "icu/chartevents.csv.gz", "icu/inputevents.csv.gz"]


def fetch(rel: str, dest: Path, tries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(BASE + rel, headers={"User-Agent": "icu-predict/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            with gzip.open(dest) as g:   # verify it's a real gzip, not an error page
                g.read(64)
            return
        except Exception as e:
            print(f"  {rel}: attempt {attempt} failed ({e})")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {rel}")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/mimic-iv-demo")
    local = os.environ.get("MIMIC_DEMO_LOCAL")
    try:
        for rel in FILES:
            dest = out / rel
            if local:
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = Path(local) / rel
                if src.exists():
                    shutil.copy(src, dest)
                else:   # local folder holds unzipped .csv files (macOS unzips automatically)
                    shutil.copy(Path(local) / rel.removesuffix(".gz"), dest.with_suffix(""))
                print(f"  copied {rel}")
            else:
                print(f"  downloading {rel} ...")
                fetch(rel, dest)
    except Exception as e:
        print(f"[fetch] failed: {e}")
        return 1
    print(f"[fetch] MIMIC-IV demo ready in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
