# PesaGuard scoring API
FROM python:3.12-slim

WORKDIR /app

# libgomp1 provides libgomp.so.1, the OpenMP runtime that the XGBoost native
# library loads at import time. python:*-slim does not ship it, so install it
# explicitly rather than relying on the wheel to vendor a copy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first to maximize Docker layer caching
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code + trained model artifacts
COPY src/ src/
COPY models/ models/

# Holds the runtime SQLite database when SUPABASE_DB_URL is not set
RUN mkdir -p data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
