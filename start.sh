#!/usr/bin/env bash
# Full-stack entrypoint for the Hugging Face Space:
# starts the FastAPI scoring API in the background, waits for it to become
# healthy, then launches the Streamlit analyst console on the HF Spaces port.
set -e

python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 &

echo "Waiting for scoring API to become healthy..."
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "Scoring API ready."
    break
  fi
  sleep 1
done

exec streamlit run src/dashboard.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
