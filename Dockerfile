# ICU-PREDICT — API + hospital UI in one container.
#   docker compose up --build          (see README for MIMIC-IV)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp1: OpenMP runtime needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Optional GRU benchmark: docker build --build-arg WITH_GRU=true .
ARG WITH_GRU=false
RUN if [ "$WITH_GRU" = "true" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi

COPY . .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
ENTRYPOINT ["./docker-entrypoint.sh"]
