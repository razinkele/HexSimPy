import os
import numpy as np
import pandas as pd
import pytest
from salmon_ibm.agents import AgentPool
from salmon_ibm.output import OutputLogger


def test_logger_creates_file(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=3, start_tri=0)
    logger.log_step(0, pool)
    logger.close()
    assert os.path.exists(tmp_path / "tracks.csv")


def test_logger_records_correct_columns(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    logger.log_step(0, pool)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert set(df.columns) >= {"time", "agent_id", "tri_idx", "lat", "lon",
                                "ed_kJ_g", "behavior", "alive", "arrived"}


def test_logger_accumulates_steps(tmp_path):
    centroids = np.array([[55.0, 21.0]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    logger.log_step(0, pool)
    logger.log_step(1, pool)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert len(df) == 4
