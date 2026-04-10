from typing import Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class CureAiAction(Action):
    task_id: str = Field(default="task_easy")
    analysis: str = Field(default="")
    fix: str = Field(default="")
    root_cause: str = Field(default="")
    done: bool = Field(default=False)


class CureAiReward(BaseModel):
    analysis_score: float = Field(default=1e-3, gt=0.0, lt=1.0)
    fix_score: float = Field(default=1e-3, gt=0.0, lt=1.0)
    root_cause_score: float = Field(default=1e-3, gt=0.0, lt=1.0)
    step_discount: float = Field(default=1.0 - 1e-3, gt=0.0, lt=1.0)
    unsafe_penalty: float = Field(default=1e-3, gt=0.0, lt=1.0)
    loop_penalty: float = Field(default=1e-3, gt=0.0, lt=1.0)
    total: float = Field(default=1e-3, gt=0.0, lt=1.0)


class CureAiState(BaseModel):
    task_id: str = Field(default="task_easy")
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=5, ge=1)
    total_reward: float = Field(default=0.0, ge=0.0)
    last_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = Field(default=False)


class CureAiObservation(Observation):
    task_id: str = Field(default="task_easy")
    description: str = Field(default="")
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    step: int = Field(default=0)
    max_steps: int = Field(default=5)
    reward: float = Field(default=1e-3, gt=0.0, lt=1.0)
    done: bool = Field(default=False)
    message: str = Field(default="")
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
