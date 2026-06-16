import torch

from uav_brl.agents.networks import DQNNetwork
from uav_brl.envs.uav_grid_env import UAVGridEnv
from uav_brl.utils.device import get_device


def main():
    device = get_device()
    print("Device:", device)

    env = UAVGridEnv(seed=0)
    state = env.reset()

    model = DQNNetwork(
        in_channels=4,
        grid_size=20,
        feature_dim=64,
        n_actions=env.n_joint_actions,
    ).to(device)

    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(x)

    print("Input shape:", x.shape)
    print("Q-values shape:", q_values.shape)
    print("Expected actions:", env.n_joint_actions)


if __name__ == "__main__":
    main()
