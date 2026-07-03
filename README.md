---
title: PesaGuard Fraud Detection
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PesaGuard — Real-Time Mobile Money Fraud Detection (Live Demo)

This Hugging Face Space runs the **full stack**: a FastAPI scoring API plus the
Streamlit analyst console, in a single always-on container.

- **Source code, architecture & benchmarks:** https://github.com/Lucas-Maingi/PesaGuard
- **What it does:** scores mobile-money transactions in real time using a hybrid
  XGBoost + Isolation Forest ensemble, with SHAP explanations for each alert.

> This is the deployment branch. The README with full documentation, honest
> benchmarks, and limitations lives on the [`main` branch on GitHub](https://github.com/Lucas-Maingi/PesaGuard).
