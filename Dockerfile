# PesaGuard — full-stack demo image for Hugging Face Spaces.
# Runs the FastAPI scoring API (localhost:8000) AND the Streamlit analyst
# console (public port 7860) in a single container via start.sh.
#
# NOTE: This is the Space/demo image and lives only on the `huggingface`
# branch. The lean API-only image used by CI lives on `main`.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by XGBoost. curl: used by the healthcheck loop.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Full dependencies (dashboard + API + model libraries)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/paysim_test_sample.csv data/paysim_test_sample.csv
COPY start.sh .

# Writable locations for the SQLite fallback DB and Streamlit's home dir
# (Hugging Face may run the container as a non-root user).
RUN mkdir -p data && chmod -R 777 data && chmod +x start.sh
ENV HOME=/tmp
ENV API_URL=http://127.0.0.1:8000

EXPOSE 7860

CMD ["./start.sh"]
