FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && python -m pip install \
      torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install \
      numpy==1.26.4 \
      sentence-transformers==3.0.1 \
      transformers==4.43.3 \
      xgboost==2.1.1 \
      scikit-learn==1.5.2 \
      pandas==2.2.2 \
      matplotlib==3.9.2 \
      joblib==1.4.2 \
      fastapi==0.115.0 \
      "uvicorn[standard]==0.30.6"


COPY app/ ./app/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY requirements.txt ./  # если есть — пусть лежит рядом (не обязателен)

RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
