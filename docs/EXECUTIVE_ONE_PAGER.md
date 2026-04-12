# Cure AI Executive One-Pager

## What It Is
Cure AI is an OpenEnv-compatible incident-response environment for evaluating agent quality in realistic production outages.

## Why It Matters
Most benchmark tasks do not capture the real constraints of SRE operations:
- noisy telemetry
- ambiguous symptoms
- high-cost unsafe actions
- urgency under limited step budget

Cure AI addresses this gap with production-inspired incidents and safety-aware scoring.

## Core Design
- Three deterministic incident tasks with escalating difficulty:
  - task_easy: database connection pool exhaustion
  - task_medium: cache misconfiguration latency cascade
  - task_hard: JWT/auth outage with overlapping failure signals
- Typed action schema:
  - analysis
  - fix
  - root_cause
  - done
- Rich observation payload:
  - description
  - logs
  - metrics
  - feedback message
  - reward breakdown

## Evaluation Philosophy
- Deterministic and reproducible task progression.
- Safety-aware grading (penalizes destructive remediations).
- Strict score-contract robustness for reliable benchmarking pipelines.

## Reliability and Compliance
- OpenEnv validation-compatible setup.
- Dockerized FastAPI deployment path.
- Hugging Face Space endpoint support.
- Parser-safe inference output contract for evaluator interoperability.

## Real-World Utility
Cure AI can be used to evaluate and compare incident-response agents for:
- diagnosis precision
- remediation safety
- root-cause labeling quality

## Judge Summary
Cure AI is a practical, reproducible, safety-aware environment that targets a high-value operational domain and enables apples-to-apples agent evaluation.
