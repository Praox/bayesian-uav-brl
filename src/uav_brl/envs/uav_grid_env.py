import numpy as np


class UAVGridEnv:
    """
    Simple 20x20 multi-UAV environment.

    State:
        shape = (4, grid_size, grid_size)

        channel 0: UAV position map
        channel 1: Bayesian belief map over target locations
        channel 2: visited map
        channel 3: uncertainty map

    Action:
        joint action for n_uavs.
        Each UAV has 5 actions:
            0 = stay
            1 = up
            2 = down
            3 = left
            4 = right

        For 3 UAVs:
            total joint actions = 5^3 = 125
    """

    def __init__(
        self,
        grid_size=20,
        n_uavs=3,
        n_targets=4,
        max_steps=100,
        sensor_range=2,
        detection_probability=1.0,
        seed=None,
    ):
        self.grid_size = grid_size
        self.n_uavs = n_uavs
        self.n_targets = n_targets
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.detection_probability = detection_probability

        self.n_actions_per_uav = 5
        self.n_joint_actions = self.n_actions_per_uav ** self.n_uavs

        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.uav_positions = []
        self.target_positions = []
        self.detected_targets = set()
        self.visited_map = None
        self.belief_map = None

    def reset(self):
        self.t = 0

        self.uav_positions = self._sample_unique_positions(self.n_uavs)
        self.target_positions = self._sample_unique_positions(self.n_targets)

        self.detected_targets = set()

        self.visited_map = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # Belief map: expected number of targets per cell.
        # Sum of belief map = number of targets.
        self.belief_map = np.ones(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        self.belief_map *= self.n_targets / (self.grid_size * self.grid_size)

        return self._get_state()

    def step(self, action_id):
        self.t += 1

        actions = self.decode_action(action_id)

        self.uav_positions = [
            self._move_uav(pos, act)
            for pos, act in zip(self.uav_positions, actions)
        ]

        reward = self._observe_and_update_belief()

        # Small time penalty to encourage efficient trajectories.
        reward -= 0.05

        # Small penalty if UAVs overlap exactly.
        reward -= self._overlap_penalty()

        done = False

        if self.t >= self.max_steps:
            done = True

        if len(self.detected_targets) == self.n_targets:
            done = True
            reward += 20.0

        info = {
            "step": self.t,
            "detected_targets": len(self.detected_targets),
            "uav_positions": list(self.uav_positions),
            "target_positions": list(self.target_positions),
            "belief_sum": float(self.belief_map.sum()),
        }

        return self._get_state(), float(reward), done, info

    def decode_action(self, action_id):
        if action_id < 0 or action_id >= self.n_joint_actions:
            raise ValueError(
                f"Invalid action_id={action_id}. "
                f"Expected value in [0, {self.n_joint_actions - 1}]"
            )

        actions = []
        value = action_id

        for _ in range(self.n_uavs):
            actions.append(value % self.n_actions_per_uav)
            value //= self.n_actions_per_uav

        return actions

    def sample_random_action(self):
        return int(self.rng.integers(0, self.n_joint_actions))

    def _sample_unique_positions(self, n):
        positions = set()

        while len(positions) < n:
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            positions.add((x, y))

        return list(positions)

    def _move_uav(self, pos, action):
        x, y = pos

        if action == 0:
            pass
        elif action == 1:
            x -= 1
        elif action == 2:
            x += 1
        elif action == 3:
            y -= 1
        elif action == 4:
            y += 1
        else:
            raise ValueError(f"Invalid UAV action: {action}")

        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))

        return (x, y)

    def _observe_and_update_belief(self):
        reward = 0.0

        for ux, uy in self.uav_positions:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    dist = abs(ux - x) + abs(uy - y)

                    if dist <= self.sensor_range:
                        self.visited_map[x, y] = 1.0

                        if (x, y) in self.target_positions:
                            target_id = self.target_positions.index((x, y))

                            detected = self.rng.random() < self.detection_probability

                            if detected:
                                if target_id not in self.detected_targets:
                                    self.detected_targets.add(target_id)
                                    reward += 10.0

                                # High belief at detected target cell.
                                self.belief_map[x, y] = 1.0
                        else:
                            # If observed and no target, reduce belief.
                            self.belief_map[x, y] *= 0.2

        self._normalize_belief_map()

        return reward

    def _normalize_belief_map(self):
        self.belief_map = np.clip(self.belief_map, 1e-6, 1.0)
        total = self.belief_map.sum()

        if total <= 0:
            self.belief_map[:] = self.n_targets / (self.grid_size * self.grid_size)
        else:
            self.belief_map *= self.n_targets / total

    def _overlap_penalty(self):
        unique_positions = set(self.uav_positions)
        n_overlaps = self.n_uavs - len(unique_positions)
        return 0.5 * n_overlaps

    def _get_state(self):
        uav_map = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        for x, y in self.uav_positions:
            uav_map[x, y] = 1.0

        uncertainty_map = 1.0 - self.visited_map

        state = np.stack(
            [
                uav_map,
                self.belief_map,
                self.visited_map,
                uncertainty_map,
            ],
            axis=0,
        )

        return state.astype(np.float32)
