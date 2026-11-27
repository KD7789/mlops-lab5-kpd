import pytest

from shopper_segments import modeling, settings


def test_candidate_grid():
    grid = modeling.candidate_grid(["kmeans", "gmm"], 4, 8)
    assert len(grid) == 10
    assert grid[0] == {"algorithm": "kmeans", "k": 4}
    assert grid[-1] == {"algorithm": "gmm", "k": 8}
    for bad in [([], 4, 8), (["svm"], 4, 8), (["kmeans"], 5, 4), (["kmeans"], 1, 4)]:
        with pytest.raises(ValueError):
            modeling.candidate_grid(*bad)


@pytest.mark.parametrize("algorithm", settings.ALGORITHMS)
def test_fit_candidate_metrics(train_sample, algorithm):
    pipeline, metrics = modeling.fit_candidate(train_sample, algorithm, 4)
    assert metrics["algorithm"] == algorithm and metrics["k"] == 4
    assert -1 <= metrics["silhouette"] <= 1
    assert metrics["segments_found"] == 4
    assert set(pipeline.predict(train_sample[settings.FEATURES])) == {0, 1, 2, 3}


def test_pipeline_selects_columns_by_name(train_sample):
    # The saved Pipeline takes raw columns in any order and does its own log
    # transform and scaling, so scoring code can't forget a preprocessing step.
    pipeline, _ = modeling.fit_candidate(train_sample, "kmeans", 4)
    shuffled = train_sample[list(reversed(train_sample.columns))]
    assert (pipeline.predict(shuffled) == pipeline.predict(train_sample)).all()


def test_pick_best_breaks_ties():
    results = [
        {"algorithm": "gmm", "k": 4, "silhouette": 0.30},
        {"algorithm": "kmeans", "k": 6, "silhouette": 0.35},
        {"algorithm": "gmm", "k": 5, "silhouette": 0.35},
        {"algorithm": "kmeans", "k": 5, "silhouette": 0.35},
    ]
    assert modeling.pick_best(results) == {"algorithm": "kmeans", "k": 5, "silhouette": 0.35}


def test_promotion_decision():
    best = {"silhouette": 0.36}
    assert not modeling.promotion_decision(best, None, min_silhouette=0.40)["promote"]
    assert modeling.promotion_decision(best, None, min_silhouette=0.25)["promote"]
    assert not modeling.promotion_decision(best, 0.36, min_silhouette=0.25)["promote"]
    assert modeling.promotion_decision(best, 0.30, min_silhouette=0.25)["promote"]


def test_segment_profiles(train_sample):
    pipeline, _ = modeling.fit_candidate(train_sample, "kmeans", 4)
    profiles = modeling.segment_profiles(pipeline, train_sample)
    assert [p["segment"] for p in profiles] == [0, 1, 2, 3]
    assert sum(p["sessions"] for p in profiles) == len(train_sample)
    assert sum(p["share"] for p in profiles) == pytest.approx(1, abs=1e-3)
    assert all(0 <= p["purchase_rate"] <= 1 for p in profiles)
