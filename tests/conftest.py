import sys
from pathlib import Path

import pandas as pd
import pytest

# The tests run from the repo root locally and from /opt/airflow in the
# container; in both cases the DAG code and the data sit next to tests/.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dags"))

from shopper_segments import data, settings  # noqa: E402


@pytest.fixture(scope="session")
def sessions() -> pd.DataFrame:
    return data.load_sessions(REPO / "data" / settings.SOURCE_FILE)


@pytest.fixture(scope="session")
def train_sample(sessions) -> pd.DataFrame:
    """2,000 cleaned training-month sessions: enough for quick model tests."""
    train = data.split_by_month(data.clean_sessions(sessions), settings.TRAIN_MONTHS)
    return train.sample(2000, random_state=0).reset_index(drop=True)
