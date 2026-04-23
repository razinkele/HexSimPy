import os
import numpy as np
import pandas as pd
import pytest
from salmon_ibm.agents import AgentPool
from salmon_ibm.output import OutputLogger
from salmon_ibm.population import Population


def test_logger_creates_file(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=3, start_tri=0)
    pop = Population(name="test", pool=pool)
    logger.log_step(0, pop)
    logger.close()
    assert os.path.exists(tmp_path / "tracks.csv")


def test_logger_records_correct_columns(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    logger.log_step(0, pop)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert set(df.columns) >= {"time", "agent_id", "tri_idx", "lat", "lon",
                                "ed_kJ_g", "behavior", "alive", "arrived"}


def test_logger_accumulates_steps(tmp_path):
    centroids = np.array([[55.0, 21.0]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    logger.log_step(0, pop)
    logger.log_step(1, pop)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert len(df) == 4


def test_logger_uses_stable_agent_ids_across_compact():
    """Agent IDs must survive Population.compact() so cross-step tracking works."""
    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    pop = Population(name="test", pool=pool)
    centroids = np.zeros((10, 2))
    logger = OutputLogger(path="/tmp/unused.csv", centroids=centroids)

    logger.log_step(0, pop)
    ids_t0 = logger._agent_ids[-1].copy()
    assert list(ids_t0) == [0, 1, 2]

    # Kill agent 1 and use the canonical compact path.
    pop.pool.alive[1] = False
    pop.compact()

    logger.log_step(1, pop)
    ids_t1 = logger._agent_ids[-1]
    # Surviving agents keep their ORIGINAL IDs, not relabeled to [0, 1].
    assert list(ids_t1) == [0, 2], f"Expected [0, 2], got {list(ids_t1)}"
