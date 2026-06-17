from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                state.astype(np.float32),
                int(action),
                float(reward),
                next_state.astype(np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones
