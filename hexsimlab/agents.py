import numpy as np


class AgentSim:
    def __init__(self, n_agents, grid):

        self.grid = grid

        self.pos = np.zeros((n_agents, 2), dtype=int)

    def step(self):
        moves = np.random.randint(0, 6, len(self.pos))
        dirs = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]])
        new_pos = self.pos + dirs[moves]
        # Bounds check: keep agents within grid
        valid = (new_pos[:, 0] >= 0) & (new_pos[:, 1] >= 0)
        self.pos[valid] = new_pos[valid]
