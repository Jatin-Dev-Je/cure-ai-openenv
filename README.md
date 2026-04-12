# Cure AI OpenEnv Submission

This repository contains the Cure AI OpenEnv project and deployment assets for hackathon submission.

## Repository Layout

- `cure_ai/` - main Python package, OpenEnv manifest, server, inference script, tests, and Dockerfiles
- `LICENSE` - project license

## Quick Links

- Hugging Face Space: https://huggingface.co/spaces/RocktheDev/cure_ai
- GitHub Repository: https://github.com/Jatin-Dev-Je/cure-ai-openenv

## Judging Assets

- Rubric Mapping: docs/RUBRIC_MAPPING.md
- Demo Script (5-7 min): docs/DEMO_SCRIPT.md
- Pre-Submit Checklist: docs/JUDGING_CHECKLIST.md
- Executive One-Pager: docs/EXECUTIVE_ONE_PAGER.md
- Architecture and Flow: docs/ARCHITECTURE.md
- Judge FAQ: docs/JUDGE_FAQ.md

## Local Setup

```bash
cd cure_ai
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -e .
```

## Run the Environment Locally

```bash
cd cure_ai
uv run server --host 0.0.0.0 --port 8000
```

## Run Baseline Inference

```bash
cd cure_ai
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_hf_token \
ENV_BASE_URL=http://localhost:8000 \
python inference.py
```

Notes:
- `HF_TOKEN` is required.
- `API_BASE_URL` and `MODEL_NAME` can use defaults in the script.
- `LOCAL_IMAGE_NAME` is optional and only needed for docker-image workflows.

## Validation

```bash
cd cure_ai
bash ./validate-submission.sh https://rockthedev-cure-ai.hf.space
```

Expected outcome: all 3 checks pass (Space reset, Docker build, `openenv validate`).

## Submission URLs

Use these in the submission form:

- Hugging Face Space URL: https://huggingface.co/spaces/RocktheDev/cure_ai
- GitHub Repo URL (page URL): https://github.com/Jatin-Dev-Je/cure-ai-openenv
