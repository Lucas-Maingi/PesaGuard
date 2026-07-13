# Production Deployment Guide

How to take PesaGuard from `docker-compose up` to something an operations team could actually run. Everything here is grounded in what the code already supports; where a step requires new work, it says so.

## Database: SQLite → Supabase (Postgres)

The DB layer ([`src/db.py`](../src/db.py)) already speaks both dialects — set one environment variable and every query switches parameter style and DDL automatically:

```bash
export SUPABASE_DB_URL="postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres"
```

Setup on Supabase:

1. Create a project, copy the **connection string** (Settings → Database → URI). Use the *session pooler* URI if you deploy on a platform with many short-lived connections.
2. No manual schema step — the app creates tables on first connection (`init_db` handles both dialects).
3. Add an index once data grows: `CREATE INDEX idx_tx_sender_time ON transactions (name_orig, step DESC);` — the realtime feature pipeline's history lookup is the hot query, and this index is what keeps it flat as volume grows.

Unset the variable and everything falls back to local SQLite (`data/pesaguard.db`) — that's the demo mode, and it's the same code path minus the connection.

## Model serving

The three artifacts (`feature_pipeline.joblib`, `xgb_model.joblib`, `iforest_model.joblib`) load once at API startup and score in-process. Deliberate choice: no model server (TorchServe/Triton) because a 100 ms budget with an XGBoost ensemble doesn't justify a network hop to another container.

**Retraining cadence:** fraud drifts fast; monthly retrain is a reasonable floor. The pattern:

1. `python train.py` against the latest labeled export (produces new joblibs + prints the comparison metrics).
2. Ship artifacts as a versioned build (bake into the image, or mount a volume and restart) — the API has no hot-reload; a rolling restart is the deployment.
3. Gate promotion on the threshold-sensitivity table (see [model-report.md](model-report.md)) — a new model whose fraud scores cluster differently can silently invalidate the operating threshold. Check the plateau, not just AUC.

## Scaling shape

- **The API is stateless** — all state lives in the DB. Run N replicas behind any load balancer; no coordination needed.
- **The hot path** is: one indexed history query → in-process feature computation → two model predicts. At M-Pesa-scale volumes (~460 tx/sec average), a handful of API replicas and a properly indexed Postgres are sufficient; the DB is the scaling boundary, not the models.
- **Async is not used, deliberately.** Scoring is CPU-bound and short; async buys nothing over workers here. Scale with `uvicorn --workers` / replicas.

## Monitoring the thing that matters

Standard infra metrics aside, fraud systems need **score-distribution monitoring**: alert when the daily mean ensemble score or the fraction above threshold drifts beyond historical bands. Score drift is your earliest signal of (a) a new fraud pattern, or (b) upstream data breakage — and it fires *before* chargebacks arrive with labels. The `transactions` table already stores every score; a scheduled query + alert is enough to start.

## Deployment targets that fit

| Target | Fit |
|---|---|
| **Hugging Face Space** (current live demo) | Demo only — sleeps, no SLA. |
| **Render / Railway** | Right size for a pilot: Docker-native, managed Postgres or Supabase alongside, ~$15–25/mo. |
| **A VPS + docker-compose** | The compose file in the repo is production-shaped (separate API and dashboard services); add Caddy for TLS and it's a legitimate small deployment. |
| **Kubernetes** | Not until an operator integration demands it. Nothing in the architecture blocks it (stateless API, external DB), but nothing requires it either. |

## What going live with a real operator adds (not built)

- **Authentication on the scoring API** — currently open; a pilot needs at minimum API-key auth and per-client rate limits.
- **PII handling** — PaySim IDs are synthetic; real MSISDNs mean encryption at rest, masked logs, and a data-retention policy before any regulator asks.
- **Feedback loop plumbing** — the `/feedback` endpoint exists; wiring it to the analysts' case-management outcome (confirmed fraud / false alarm) is what makes monthly retrains actually improve.
