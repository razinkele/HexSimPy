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


def test_logger_preallocated_path_matches_list_path():
    """Preallocated and list-append paths must produce identical DataFrames."""
    pool = AgentPool(n=5, start_tri=0, rng_seed=0)
    pop = Population(name="test", pool=pool)
    centroids = np.random.RandomState(0).rand(10, 2)

    list_logger = OutputLogger(path="/tmp/a.csv", centroids=centroids)
    prealloc_logger = OutputLogger(
        path="/tmp/b.csv", centroids=centroids, max_steps=3, max_agents=5
    )
    for t in range(3):
        list_logger.log_step(t, pop)
        prealloc_logger.log_step(t, pop)

    df_list = list_logger.to_dataframe().sort_values(["time", "agent_id"]).reset_index(drop=True)
    df_pre = prealloc_logger.to_dataframe().sort_values(["time", "agent_id"]).reset_index(drop=True)
    assert len(df_list) == len(df_pre) == 15
    for col in ["time", "agent_id", "tri_idx", "behavior", "alive", "arrived"]:
        np.testing.assert_array_equal(df_list[col].values, df_pre[col].values, err_msg=col)
    for col in ["lat", "lon", "ed_kJ_g"]:
        np.testing.assert_allclose(df_list[col].values, df_pre[col].values, rtol=1e-12, err_msg=col)


def test_logger_preallocated_raises_on_overflow():
    """Exceeding max_steps or max_agents must raise, not corrupt."""
    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    pop = Population(name="test", pool=pool)
    centroids = np.zeros((10, 2))
    logger = OutputLogger(path="/tmp/b.csv", centroids=centroids, max_steps=2, max_agents=3)
    logger.log_step(0, pop)
    logger.log_step(1, pop)
    with pytest.raises(ValueError, match="max_steps"):
        logger.log_step(2, pop)


def test_logger_preallocated_empty_returns_empty_df():
    pool = AgentPool(n=1, start_tri=0, rng_seed=0)
    centroids = np.zeros((10, 2))
    logger = OutputLogger(path="/tmp/c.csv", centroids=centroids, max_steps=5, max_agents=1)
    df = logger.to_dataframe()
    assert len(df) == 0
    assert "time" in df.columns


def test_outputlogger_serialises_natal_reach_id(tmp_path):
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pool.natal_reach_id[:] = np.array([5, 6, 7], dtype=np.int8)
    pool.exit_branch_id[:] = np.array([8, -1, 9], dtype=np.int8)
    pop = Population(name="test", pool=pool)
    logger = OutputLogger(path=str(tmp_path / "out.csv"),
                          centroids=np.zeros((10, 2)))
    logger.log_step(t=0, population=pop)
    df = logger.to_dataframe()
    assert "natal_reach_id" in df.columns
    assert df["natal_reach_id"].tolist() == [5, 6, 7]


def test_outputlogger_serialises_exit_branch_id(tmp_path):
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger
    import numpy as np
    pool = AgentPool(n=2, start_tri=np.array([0, 1]))
    pool.exit_branch_id[:] = np.array([4, -1], dtype=np.int8)
    pop = Population(name="test", pool=pool)
    logger = OutputLogger(path=str(tmp_path / "out.csv"),
                          centroids=np.zeros((10, 2)))
    logger.log_step(t=0, population=pop)
    df = logger.to_dataframe()
    assert "exit_branch_id" in df.columns
    assert df["exit_branch_id"].tolist() == [4, -1]
