import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    """
    CNN feature extractor.

    Input:
        batch of states with shape:
            (batch_size, 4, 20, 20)

    Output:
        feature vector phi(s) with shape:
            (batch_size, feature_dim)
    """

    def __init__(self, in_channels=4, grid_size=20, feature_dim=64):
        super().__init__()

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        flattened_dim = 32 * grid_size * grid_size

        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        phi = F.relu(self.fc2(x))

        return phi


class DQNHead(nn.Module):
    """
    Standard DQN head.

    This is not Bayesian yet.
    We keep it because the clean path is:
        1. random baseline
        2. DQN baseline
        3. BDQN
    """

    def __init__(self, feature_dim=64, n_actions=125):
        super().__init__()
        self.q = nn.Linear(feature_dim, n_actions)

    def forward(self, phi):
        return self.q(phi)


class DQNNetwork(nn.Module):
    def __init__(
        self,
        in_channels=4,
        grid_size=20,
        feature_dim=64,
        n_actions=125,
    ):
        super().__init__()

        self.feature_net = FeatureNet(
            in_channels=in_channels,
            grid_size=grid_size,
            feature_dim=feature_dim,
        )

        self.head = DQNHead(
            feature_dim=feature_dim,
            n_actions=n_actions,
        )

    def forward(self, x):
        phi = self.feature_net(x)
        q_values = self.head(phi)
        return q_values
