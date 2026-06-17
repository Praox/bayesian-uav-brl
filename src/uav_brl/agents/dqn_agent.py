import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from uav_brl.agents.networks import DQNNetwork
from uav_brl.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_channels,
        grid_size,
        n_actions,
        device,
        feature_dim=64,
        replay_capacity=50_000,
        batch_size=64,
        gamma=0.95,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        target_update_freq=500,
    ):
        self.n_actions = n_actions
        self.device = device

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.total_steps = 0

        self.policy_net = DQNNetwork(
            in_channels=state_channels,
            grid_size=grid_size,
            feature_dim=feature_dim,
            n_actions=n_actions,
        ).to(device)

        self.target_net = DQNNetwork(
            in_channels=state_channels,
            grid_size=grid_size,
            feature_dim=feature_dim,
            n_actions=n_actions,
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            device=device,
        )

    def epsilon(self):
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, state):
        self.total_steps += 1

        eps = self.epsilon()

        if random.random() < eps:
            return random.randrange(self.n_actions)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]

            targets = rewards + self.gamma * (1.0 - dones) * max_next_q_values

        loss = F.smooth_l1_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
