---
title: Cure AI - Incident Response RL Environment
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
---

### Overview

**Cure AI** is an OpenEnv-compatible reinforcement learning environment that simulates **on'call incident response** for production systems.  
At the start of an episode, the agent receives an incident with **description, logs, and metrics**.  
On each step, the agent responds with:
- **analysis**: diagnosis of whats going wrong
- **fix**: a safe remediation plan
- **root_cause**: a short label for the true cause

The environment returns a **shaped reward in \[0.0, 1.0\]** based on how well the response matches the ground'truth failure mode and remediation, with penalties for unsafe actions.

### Tasks

- **Easy " DB connection pool exhaustion (`task_easy`)**
  - Symptoms: connection timeouts, pool exhaustion, high error rate.
  - Goal: identify the database connection pool issue and propose a safe fix (pool tuning, backoff, etc.).

- **Medium " Cache disabled cascade (`task_medium`)**
  - Symptoms: cache disabled flag, cache misses, fallback to origin, elevated latency.
  - Goal: attribute the latency spike to cache misconfiguration and restore healthy caching behavior.

- **Hard " JWT misconfiguration with noise (`task_hard`)**
  - Symptoms: intermittent 401s, JWT signature mismatch, clock skew, noisy user reports.
  - Goal: trace the outage to an auth/JWT issue (e.g. secret mismatch or time skew) and propose a safe recovery plan.

Each task has a **deterministic grader** that scores analysis, fix, and root cause, then applies a **step'based discount** so faster, more accurate resolutions get higher reward.

### Action Space (`CureAiAction`)

- **`task_id: str`**  
  Identifier of the current task (`"task_easy" | "task_medium" | "task_hard"`).  
  Primarily used for bookkeeping; the environment samples the task on `reset()`.

- **`analysis: str`**  
  Free'form reasoning about what is going wrong. Grader looks for task'specific signals (e.g. *connection pool*, *cache disabled*, *JWT secret*).

- **`fix: str`**  
  Proposed remediation plan. Grader rewards safe, targeted fixes and penalizes destructive operations (e.g. *drop database*).

- **`root_cause: str`**  
  Short label for the main cause (e.g. `"db_pool"`, `"cache_disabled"`, `"jwt_secret_mismatch"`).

- **`done: bool`**  
  Set to `True` when the agent believes the incident is fully handled. The environment will also terminate when `max_steps` is reached.

### Observation Space (`CureAiObservation`)

- **`task_id: str`** " Current task identifier.  
- **`description: str`** " High'level incident description.  
- **`logs: List[str]`** " Key log lines relevant to the incident (e.g. timeouts, cache misses, JWT errors).  
- **`metrics: Dict[str, float]`** " Aggregated metrics such as `error_rate` and `latency`.  
- **`step: int`** " Current step number (starts at 0 after `reset`).  
- **`max_steps: int`** " Maximum number of allowed steps in the episode.  
- **`reward: float`** " Shaped reward for this step in \[0.0, 1.0\].  
- **`done: bool`** " Whether the episode has terminated.  
- **`message: str`** " Grader feedback for the agent (e.g. Correctly identified database layer.).

### Reward & Grading

On each `step(action)`, the environment:
- Scores:
  - **analysis** (up to ~0.4) for correctly focusing on the right subsystem and symptoms.
  - **fix** (up to ~0.4) for proposing safe, task'appropriate remediation.
  - **root_cause** (up to ~0.2) for matching the expected cause label.
- Applies a **step discount** (e.g. `0.9^(step-1)`) so earlier resolutions get more credit.
- Applies penalties for unsafe patterns (e.g. `drop database`, destructive commands), then clamps the final reward to \[0.0, 1.0\].

This produces **dense, step'wise feedback**, not just a terminal score.

The environment implements the full API contract expected by OpenEnv:
- `reset()` returns the first observation of the episode.
- `step(action)` returns `(observation, reward, done, info)`.
- `state()` returns current task id, step counters, and reward totals.

Task sequencing is deterministic for reproducibility:
- Episode 1: `task_easy`
- Episode 2: `task_medium`
- Episode 3: `task_hard`
- Then repeats.

### Quick Start (Local)

Install dependencies and run the server locally:

```bash
cd cure_ai
pip install -e .
uv run server --host 0.0.0.0 --port 8000
```

Then connect with the typed client:

```python
from cure_ai import CureAiAction, CureAiEnv

with CureAiEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(result.observation.description)

    action = CureAiAction(
        analysis="Database connection pool appears exhausted due to too many clients.",
        fix="Increase connection pool size safely and add exponential backoff.",
        root_cause="db_pool",
        done=True,
    )
    step_result = env.step(action)
    print(step_result.reward, step_result.observation.message)
```

### Docker & OpenEnv

- **OpenEnv manifest**: `openenv.yaml` points to the FastAPI app:
  - `runtime: fastapi`
  - `app: server.app:app`
  - `port: 8000`
- **Dockerfile** under `server/`:
  - Uses `openenv-base` as the base image.
  - Installs dependencies via `uv sync`.
  - Exposes the environment via `uvicorn server.app:app`.
  - Includes a `/health` check provided by `create_app`.

Build and run via:

```bash
cd cure_ai
docker build -t cure-ai:latest -f server/Dockerfile .
docker run -p 8000:8000 cure-ai:latest
```

### Baseline Inference

The file `inference.py` provides a **baseline agent** using the OpenAI Python SDK pointed at the **Hugging Face router**:

- Reads configuration from environment variables:
  - `HF_TOKEN` " router API key.
  - `API_BASE_URL` " router base URL (not `api.openai.com`).
  - `MODEL_NAME` " model identifier to query via the router.
- Uses `CureAiEnv` to:
  - Reset the environment.
  - Run up to `max_steps` per episode.
  - Call the model each step to generate `analysis`, `fix`, `root_cause`, `done`.
  - Accumulate rewards across **easy**, **medium**, and **hard** tasks in deterministic order.
- Emits structured logs for evaluators:
  - `[START] task_id=... model=... max_steps=...`
  - `[STEP] task_id=... step=... reward=... done=... root_cause=...`
  - `[END] task_id=... total_reward=... steps=... done=...`
- Writes a `results.json` file with:
  - Per'task total rewards and steps.
  - Overall mean reward.
  - Model name and configuration.

### Baseline Performance (Local Reproducible)

Run baseline inference with the OpenAI client configured via required variables:

```bash
cd cure_ai
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_hf_token \
ENV_BASE_URL=http://localhost:8000 \
python inference.py
```

`HF_TOKEN` is mandatory. `API_BASE_URL` and `MODEL_NAME` have defaults in `inference.py`, but you can override them explicitly.

Latest local run (`run_id=20260408105351`) produced:
- `task_easy`: total_reward `3.6856`, steps `5`
- `task_medium`: total_reward `3.6856`, steps `5`
- `task_hard`: total_reward `2.4571`, steps `5`
- mean_reward: `3.2761`

This script serves as the **reproducible baseline** for hackathon evaluation.

### Project Structure

```
cure_ai/
""" __init__.py
""" README.md
""" client.py
""" inference.py
""" models.py
""" openenv.yaml
""" pyproject.toml
""" uv.lock
"""" server/
    """ __init__.py
    """ app.py
    """ cure_ai_environment.py
    """ requirements.txt
    """" Dockerfile
```

The environment is designed to **pass `openenv validate`**, build as a **Hugging Face Space**, and act as a **realistic incident'response benchmark** for agentic RL.
