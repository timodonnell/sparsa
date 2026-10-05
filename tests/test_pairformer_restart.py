from scripts.restart_pairformer_at_boundary import checkpoint_ready


def observation(step=100):
    return {
        "complete": {
            "full_validation_proteins": 97,
            "rollouts": 100,
            "held_out_used": False,
        },
        "latest": {"step": step},
        "checkpoint_exists": True,
        "latest_age": 5,
        "complete_age": 3600,
        "saved_log_step": step,
    }


def test_waits_for_complete_validation_and_durable_fresh_checkpoint():
    state = observation()
    assert checkpoint_ready(state, 100)
    for key, value in [
        ("complete", None),
        ("checkpoint_exists", False),
        ("latest_age", 31),
        ("saved_log_step", 10),
    ]:
        assert not checkpoint_ready({**state, key: value}, 100)
    assert not checkpoint_ready(state, 200)


def test_validation_boundary_can_reuse_the_unchanged_step_four_checkpoint():
    state = observation(4)
    state.update(latest_age=10000, complete_age=5, saved_log_step=4)
    assert checkpoint_ready(state, 4)
    assert not checkpoint_ready({**state, "complete_age": 31}, 4)
    assert not checkpoint_ready(state, 100)


def test_rejects_incomplete_or_wrong_validation():
    state = observation()
    for key, value in [
        ("full_validation_proteins", 96),
        ("rollouts", 99),
        ("held_out_used", True),
    ]:
        changed = {**state, "complete": {**state["complete"], key: value}}
        assert not checkpoint_ready(changed, 100)
