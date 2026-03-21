"""Shared mock/helper classes for salmon_ibm tests.

These are plain Python classes (not pytest fixtures) that can be imported
directly by test modules:

    from tests.helpers import MockPopulation
"""

import numpy as np


class MockPopulation:
    """Minimal population mock for unit tests.

    Parameters
    ----------
    n:
        Number of agents.
    positions:
        Cell indices for each agent.  Defaults to all zeros.
    alive:
        Boolean alive mask.  Defaults to all True.
    """

    def __init__(self, n, positions=None, alive=None):
        self.n = n
        self.tri_idx = (
            np.zeros(n, dtype=np.int64)
            if positions is None
            else np.array(positions, dtype=np.int64)
        )
        self.alive = (
            np.ones(n, dtype=bool) if alive is None else np.array(alive, dtype=bool)
        )
        self.arrived = np.zeros(n, dtype=bool)
        self.group_id = np.full(n, -1, dtype=np.int32)
