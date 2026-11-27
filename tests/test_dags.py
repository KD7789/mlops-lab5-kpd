"""DAG integrity tests: the DAG files import without errors and are wired as intended."""

from pathlib import Path

import pytest

try:  # where the file-parsing DagBag lives in recent Airflow 3 releases
    from airflow.dag_processing.dagbag import DagBag
except ImportError:
    from airflow.models.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR))


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_training_dag_wiring(dagbag):
    dag = dagbag.dags["shopper_segments_training"]
    downstream = {t.task_id: set(t.downstream_task_ids) for t in dag.tasks}
    assert downstream["check_data"] == {"prepare_training_data", "quarantine_batch"}
    assert downstream["promotion.check_gate"] == {"promotion.approve_model", "promotion.keep_champion"}
    assert downstream["promotion.approve_model"] == {"promotion.promote_model"}
    assert downstream["promotion.promote_model"] == {"write_run_summary"}
    assert downstream["promotion.keep_champion"] == {"write_run_summary"}
    assert dag.get_task("write_run_summary").trigger_rule == "none_failed"


def test_training_dag_settings(dagbag):
    dag = dagbag.dags["shopper_segments_training"]
    assert dag.max_active_runs == 1
    assert dag.catchup is False
    assert set(dag.params) == {"k_min", "k_max", "algorithms", "min_silhouette", "source_file"}
    assert dag.get_task("model_search.train_candidate").max_active_tis_per_dagrun == 3
    assert [a.name for a in dag.get_task("promotion.promote_model").outlets] == ["shopper_segment_champion"]
