from uav_brl.envs.uav_grid_env import UAVGridEnv
from uav_brl.agents.random_agent import RandomAgent


def run_episode(env, agent, render_logs=False):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(state, env=env)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        if render_logs:
            print(
                f"step={info['step']:03d} "
                f"reward={reward:6.2f} "
                f"detected={info['detected_targets']} "
                f"uavs={info['uav_positions']}"
            )

    return total_reward, info["detected_targets"]


def main():
    env = UAVGridEnv(
        grid_size=20,
        n_uavs=3,
        n_targets=4,
        max_steps=100,
        sensor_range=2,
        detection_probability=1.0,
        seed=42,
    )

    agent = RandomAgent(n_actions=env.n_joint_actions)

    n_episodes = 20
    rewards = []
    detected_counts = []

    for episode in range(n_episodes):
        total_reward, detected = run_episode(env, agent, render_logs=False)

        rewards.append(total_reward)
        detected_counts.append(detected)

        print(
            f"episode={episode:03d} "
            f"total_reward={total_reward:8.2f} "
            f"detected_targets={detected}/4"
        )

    print()
    print("Random baseline summary")
    print("-----------------------")
    print(f"mean reward: {sum(rewards) / len(rewards):.2f}")
    print(f"mean detected targets: {sum(detected_counts) / len(detected_counts):.2f}/4")


if __name__ == "__main__":
    main()
