from cure_ai.models import CureAiAction
from cure_ai.server.cure_ai_environment import CureAiEnvironment


def test_task_cycle_order() -> None:
    env = CureAiEnvironment()
    t1 = env.reset().task_id
    t2 = env.reset().task_id
    t3 = env.reset().task_id
    cycle = ["task_easy", "task_medium", "task_hard"]
    start_idx = cycle.index(t1)
    assert [t1, t2, t3] == [
        cycle[start_idx],
        cycle[(start_idx + 1) % 3],
        cycle[(start_idx + 2) % 3],
    ]


def test_reward_is_bounded() -> None:
    env = CureAiEnvironment()
    env.reset()
    action = CureAiAction(
        task_id="task_easy",
        analysis="database pool timeout",
        fix="increase pool and add retry backoff",
        root_cause="db_pool",
        done=False,
    )
    for _ in range(5):
        step_obs = env.step(action)
        assert 0.0 <= float(step_obs.reward) <= 1.0
        if step_obs.done:
            break


def test_state_shape() -> None:
    env = CureAiEnvironment()
    env.reset()
    state = env.state
    state_dict = state.model_dump() if hasattr(state, "model_dump") else state.dict()
    assert set(state_dict.keys()) == {"episode_id", "step_count"}
