from uav_brl.envs.uav_grid_env import UAVGridEnv


def test_reset_state_shape():
    env = UAVGridEnv(seed=0)
    state = env.reset()

    assert state.shape == (4, 20, 20)


def test_action_space_size():
    env = UAVGridEnv(n_uavs=3)
    assert env.n_joint_actions == 125


def test_step_runs():
    env = UAVGridEnv(seed=0)
    state = env.reset()

    action = env.sample_random_action()
    next_state, reward, done, info = env.step(action)

    assert next_state.shape == (4, 20, 20)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "detected_targets" in info


def test_belief_sum_close_to_number_of_targets():
    env = UAVGridEnv(seed=0)
    env.reset()
    action = env.sample_random_action()
    env.step(action)

    assert abs(env.belief_map.sum() - env.n_targets) < 1e-4
