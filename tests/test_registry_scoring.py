import pytest

from shopper_segments import data, modeling, registry, scoring, settings


def test_safe_name():
    name = registry.safe_name("manual__2026-11-27T01:34:20.640304+00:00")
    assert name == "manual__2026-11-27T01_34_20.640304_00_00"


def test_promote_keeps_history(tmp_path, train_sample):
    assert registry.read_champion(tmp_path) is None
    pipeline, metrics = modeling.fit_candidate(train_sample, "kmeans", 4)
    model_path = registry.run_dir(tmp_path, "run1") / "candidates" / "kmeans_k4.joblib"
    registry.save_model(pipeline, model_path)

    first = registry.promote(tmp_path, model_path, {**metrics, "run_id": "run1"})
    second = registry.promote(tmp_path, model_path, {**metrics, "run_id": "run2"})

    assert (first["version"], second["version"]) == ("v1", "v2")
    assert registry.read_champion(tmp_path)["run_id"] == "run2"
    assert (tmp_path / "models" / "history" / "v1" / "model.joblib").exists()
    model, champion = registry.load_champion_model(tmp_path)
    assert champion["version"] == "v2"
    assert len(model.predict(train_sample[settings.FEATURES])) == len(train_sample)


def test_summarize_new_sessions(sessions, train_sample):
    pipeline, metrics = modeling.fit_candidate(train_sample, "kmeans", 4)
    champion = {**metrics, "version": "v1", "profiles": modeling.segment_profiles(pipeline, train_sample)}
    new = data.split_by_month(data.clean_sessions(sessions), settings.NEW_MONTHS)

    summary = scoring.summarize(scoring.score_sessions(pipeline, new), champion)

    assert summary["sessions"] == len(new) == 4688
    assert summary["months"] == ["Nov", "Dec"]
    assert [s["segment"] for s in summary["segments"]] == [0, 1, 2, 3]
    assert sum(s["sessions"] for s in summary["segments"]) == len(new)
    assert sum(s["share"] for s in summary["segments"]) == pytest.approx(1, abs=1e-3)
