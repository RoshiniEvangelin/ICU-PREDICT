# Hugging Face Spaces (Docker SDK). Trains during the build, so the app starts instantly.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 WARM_START=1
WORKDIR $HOME/app

COPY --chown=user requirements-space.txt .
RUN pip install --user -r requirements-space.txt

COPY --chown=user . .
RUN sh scripts/space_build.sh

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
