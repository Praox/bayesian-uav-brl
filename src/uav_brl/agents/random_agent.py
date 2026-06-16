class RandomAgent:
    def __init__(self, n_actions, rng=None):
        self.n_actions = n_actions
        self.rng = rng

    def act(self, state, env=None):
        if env is not None:
            return env.sample_random_action()

        if self.rng is None:
            import numpy as np
            return int(np.random.randint(0, self.n_actions))

        return int(self.rng.integers(0, self.n_actions))
