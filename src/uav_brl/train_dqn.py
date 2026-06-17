import os
from collections import deque

import numpy as np
from tqdm import trange

from uav_brl.envs.uav_grid_env import UAVGridEnv
from uav_brl.agents.dqn_agent import DQNAgent
from uav_brl.utils.device import get_device


def evaluate_agent(env, agent, n_episodes=5):
    old_epsilon_start = agent.epsilon_start
    old_epsilon_end = agent.epsilon_end

    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0

    rewards = []
    detected = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        detected.append(info["detected_targets"])

    agent.epsilon_start = old_epsilon_start
    agent.epsilon_end = old_epsilon_end

    return np.mean(rewards), np.mean(detected)


def main():
    device = get_device()
    print(f"Using device: {device}")

    env = UAVGridEnv(
        grid_size=20,
        n_uavs=3,
        n_targets=4,
        max_steps=100,
        sensor_range=2,
        detection_probability=1.0,
        seed=42,
    )

    agent = DQNAgent(
        state_channels=4,
        grid_size=20,
        n_actions=env.n_joint_actions,
        device=device,
        feature_dim=64,
        replay_capacity=50_000,
        batch_size=64,
        gamma=0.95,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        target_update_freq=500,
    )

    os.makedirs("experiments/dqn", exist_ok=True)

    n_episodes = 500
    recent_rewards = deque(maxlen=50)
    recent_detected = deque(maxlen=50)
    recent_losses = deque(maxlen=100)

    progress = trange(n_episodes, desc="Training DQN")

    for episode in progress:
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                recent_losses.append(loss)

            total_reward += reward
            state = next_state

        recent_rewards.append(total_reward)
        recent_detected.append(info["detected_targets"])

        avg_reward = np.mean(recent_rewards)
        avg_detected = np.mean(recent_detected)
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0

        progress.set_postfix(
            {
                "avg_reward": f"{avg_reward:.2f}",
                "avg_detected": f"{avg_detected:.2f}/4",
                "epsilon": f"{agent.epsilon():.3f}",
                "loss": f"{avg_loss:.4f}",
            }
        )

        if (episode + 1) % 100 == 0:
            eval_reward, eval_detected = evaluate_agent(env, agent, n_episodes=10)

            print()
            print(
                f"[Eval episode {episode + 1}] "
                f"reward={eval_reward:.2f}, "
                f"detected={eval_detected:.2f}/4"
            )

            save_path = f"experiments/dqn/dqn_episode_{episode + 1}.pt"
            agent.save(save_path)
            print(f"Saved model to {save_path}")

    agent.save("experiments/dqn/dqn_final.pt")
    print("Training finished.")
    print("Final model saved to experiments/dqn/dqn_final.pt")


if __name__ == "__main__":
    main()
