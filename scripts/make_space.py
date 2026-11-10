"""Build a folder ready to upload to a Hugging Face Space.

    python scripts/make_space.py [out_dir]            # Docker SDK   (default: ../icu-predict-space)
    python scripts/make_space.py [out_dir] --gradio   # Gradio SDK, for accounts where Docker Spaces aren't available
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gradio = "--gradio" in sys.argv
args = [a for a in sys.argv[1:] if a != "--gradio"]
out = Path(args[0]) if args else ROOT.parent / ("icu-predict-space-gradio" if gradio else "icu-predict-space")
TOP_SKIP = {"data", "outputs", ".venv", ".git", "tests", ".pytest_cache", "deploy", ".no-mimic", "docs"}


def ignore(d, names):
    top = Path(d).resolve() == ROOT
    return [n for n in names if (top and n in TOP_SKIP) or n == "__pycache__" or n.endswith((".zip", ".pyc"))]


if out.exists():
    shutil.rmtree(out)
shutil.copytree(ROOT, out, ignore=ignore)
if gradio:   # Gradio SDK runs `python space_app.py` after installing requirements.txt
    shutil.copy(ROOT / "deploy/huggingface/space_app.py", out / "space_app.py")
    shutil.copy(ROOT / "deploy/huggingface/SPACE_README_GRADIO.md", out / "README.md")
    shutil.copy(ROOT / "requirements-space.txt", out / "requirements.txt")
    (out / "requirements-space.txt").unlink()
    (out / "Dockerfile").unlink(missing_ok=True)
else:
    shutil.copy(ROOT / "deploy/huggingface/Dockerfile", out / "Dockerfile")
    shutil.copy(ROOT / "deploy/huggingface/SPACE_README.md", out / "README.md")
for f in ("docker-compose.yml", "docker-entrypoint.sh", ".dockerignore", "requirements-gru.txt", "pytest.ini"):
    (out / f).unlink(missing_ok=True)
print(f"Space folder ready: {out}")
