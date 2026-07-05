---
title: Clinical Trial Prediction Suite
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Clinical Trial Prediction Suite

Docker-ready Hugging Face Spaces package for a Streamlit-based clinical trial prediction demo. This version preserves the original app structure while replacing local torch-based model loading with hosted inference plus a heuristic fallback mode.

## Included files

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `sample_secrets.txt`

## Optional secret

Add this secret in your Space settings for better model-backed responses:

- `HF_API_KEY`

Without the secret, the app still runs using the built-in deterministic fallback logic.
