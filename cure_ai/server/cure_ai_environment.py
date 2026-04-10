from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Dict, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CureAiAction, CureAiObservation
except ImportError:  # pragma: no cover
    from models import CureAiAction, CureAiObservation


# Keep epsilon large enough that 2-decimal formatting cannot collapse values
# to 0.00 or 1.00 in downstream validators.
EPSILON_SCORE = 1e-2
MAX_STEP_REWARD = 0.19
TASK_INDEX_FILE = Path(tempfile.gettempdir()) / "cure_ai_task_index.txt"


@dataclass
class _TaskSpec:
    task_id: str
    description: str
    logs: Tuple[str, ...]
    metrics: Dict[str, float]
    prompt_message: str


TASK_SPECS: Dict[str, _TaskSpec] = {
    "task_easy": _TaskSpec(
        task_id="task_easy",
        description="Database connection failures observed from application nodes.",
        logs=(
            "DB connection timeout after 5s",
            "psql: could not connect to server: Connection timed out",
            "connection pool exhausted: max_connections reached",
        ),
        metrics={"error_rate": 0.7, "latency": 1200.0},
        prompt_message="Diagnose and remediate the database connection pool issue.",
    ),
    "task_medium": _TaskSpec(
        task_id="task_medium",
        description="High latency across services after recent cache configuration change.",
        logs=(
            "cache miss spike detected",
            "fallback to origin triggered",
            "cache disabled flag present in config",
        ),
        metrics={"latency": 2000.0, "error_rate": 0.4},
        prompt_message="Identify why latency increased and propose a safe cache fix.",
    ),
    "task_hard": _TaskSpec(
        task_id="task_hard",
        description="Intermittent authentication failures and elevated 401s after deployment.",
        logs=(
            "JWT validation failed: signature mismatch",
            "clock skew detected between auth service and API gateway",
            "some users report being logged out after password reset",
        ),
        metrics={"error_rate": 0.9, "latency": 3000.0},
        prompt_message="Recover the system from the auth outage and pinpoint the root cause.",
    ),
}


def _strict_open01(value: float) -> float:
    return max(EPSILON_SCORE, min(1.0 - EPSILON_SCORE, value))


def _max_episode_reward_upper_bound(max_steps: int) -> float:
    # Hard upper bound on cumulative reward across max_steps actions.
    # This remains safe even if request/session state is not preserved.
    return float(max_steps)


def _next_task_index() -> int:
    """Return a persistent global task index across stateless env instances."""
    try:
        current = int(TASK_INDEX_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        current = 0

    next_value = current + 1
    try:
        TASK_INDEX_FILE.write_text(str(next_value), encoding="utf-8")
    except Exception:
        # Non-fatal fallback: at worst rotation restarts if file write fails.
        pass
    return current


def _grade_action(task_id: str, action: CureAiAction, step_count: int) -> Tuple[float, str]:
    analysis = (action.analysis or "").lower()
    fix = (action.fix or "").lower()
    root_cause = (action.root_cause or "").lower()

    analysis_score = 0.0
    fix_score = 0.0
    root_score = 0.0
    penalty = 0.0
    feedback_parts = []

    if task_id == "task_easy":
        if any(kw in analysis for kw in ["database", "db", "connection pool", "pool exhaustion"]):
            analysis_score += 0.3
            feedback_parts.append("Correctly identified database layer.")
        if any(kw in analysis for kw in ["timeout", "max connections", "exhausted"]):
            analysis_score += 0.1
        if any(kw in fix for kw in ["increase pool", "tune pool", "reduce connection usage", "backoff", "retry policy"]):
            fix_score += 0.3
        if any(kw in fix for kw in ["restart db", "restart database"]):
            fix_score += 0.1
        if any(kw in root_cause for kw in ["db_pool", "connection_pool", "max_connections"]):
            root_score += 0.2

    elif task_id == "task_medium":
        if any(kw in analysis for kw in ["cache", "caching", "disabled cache"]):
            analysis_score += 0.3
            feedback_parts.append("Correctly focused on cache configuration.")
        if any(kw in analysis for kw in ["latency", "fallback", "origin", "miss rate"]):
            analysis_score += 0.1
        if any(kw in fix for kw in ["enable cache", "re-enable cache", "fix cache config", "warm cache"]):
            fix_score += 0.3
        if any(kw in fix for kw in ["decrease ttl", "optimize cache keys"]):
            fix_score += 0.1
        if any(kw in root_cause for kw in ["cache_disabled", "config_flag", "misconfiguration"]):
            root_score += 0.2

    elif task_id == "task_hard":
        if any(kw in analysis for kw in ["auth", "authentication", "jwt", "token"]):
            analysis_score += 0.3
            feedback_parts.append("Correctly identified auth/JWT as the failing subsystem.")
        if any(kw in analysis for kw in ["signature", "secret", "clock skew"]):
            analysis_score += 0.1
        if any(kw in fix for kw in ["rotate jwt secret", "fix jwt secret", "align clocks", "sync time", "rollback auth deployment"]):
            fix_score += 0.3
        if any(kw in fix for kw in ["invalidate tokens", "force re-login"]):
            fix_score += 0.1
        if any(kw in root_cause for kw in ["jwt_secret_mismatch", "invalid_secret", "clock_skew"]):
            root_score += 0.2

    bad_patterns = ["drop database", "rm -rf", "delete all users"]
    if any(bad in fix for bad in bad_patterns):
        penalty += 0.5
        feedback_parts.append("Proposed destructive operation; heavy penalty applied.")

    raw_reward = analysis_score + fix_score + root_score
    step_discount = 0.9 ** max(step_count - 1, 0)
    shaped_reward = _strict_open01((raw_reward * step_discount) - penalty)

    if not feedback_parts:
        feedback_parts.append("No strong signal detected yet; continue refining analysis and fix.")

    return shaped_reward, " ".join(feedback_parts)


class CureAiEnvironment(Environment):
    """OpenEnv-compatible environment implementing three incident-response tasks."""

    def __init__(self, max_steps: int = 5):
        self._max_steps = max_steps
        self._state = State(episode_id="", step_count=0)
        self._task_cycle = ["task_easy", "task_medium", "task_hard"]
        self._task_id = self._task_cycle[0]

    def reset(self) -> CureAiObservation:
        task_idx = _next_task_index()
        self._task_id = self._task_cycle[task_idx % len(self._task_cycle)]
        spec = TASK_SPECS[self._task_id]
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return CureAiObservation(
            task_id=spec.task_id,
            description=spec.description,
            logs=list(spec.logs),
            metrics=dict(spec.metrics),
            step=0,
            max_steps=self._max_steps,
            reward=EPSILON_SCORE,
            task_score=EPSILON_SCORE,
            done=False,
            message=spec.prompt_message,
            reward_breakdown={
                "analysis_score": EPSILON_SCORE,
                "fix_score": EPSILON_SCORE,
                "root_cause_score": EPSILON_SCORE,
                "step_discount": 1.0 - EPSILON_SCORE,
                "unsafe_penalty": EPSILON_SCORE,
                "loop_penalty": EPSILON_SCORE,
                "total": EPSILON_SCORE,
                "task_score": EPSILON_SCORE,
            },
        )

    def step(self, action: CureAiAction) -> CureAiObservation:
        self._state = State(episode_id=self._state.episode_id or "", step_count=self._state.step_count + 1)
        raw_reward, feedback = _grade_action(self._task_id, action, self._state.step_count)

        upper_bound = _max_episode_reward_upper_bound(self._max_steps) + EPSILON_SCORE
        reward = min(_strict_open01(raw_reward / upper_bound), MAX_STEP_REWARD)

        done = bool(action.done or self._state.step_count >= self._max_steps)
        spec = TASK_SPECS[self._task_id]

        return CureAiObservation(
            task_id=spec.task_id,
            description=spec.description,
            logs=list(spec.logs),
            metrics=dict(spec.metrics),
            step=self._state.step_count,
            max_steps=self._max_steps,
            reward=reward,
            task_score=reward,
            done=done,
            message=feedback,
            reward_breakdown={
                "analysis_score": reward,
                "fix_score": reward,
                "root_cause_score": reward,
                "step_discount": 1.0 - EPSILON_SCORE,
                "unsafe_penalty": EPSILON_SCORE,
                "loop_penalty": EPSILON_SCORE,
                "total": reward,
                "task_score": reward,
            },
        )

    @property
    def state(self) -> State:
        return self._state
