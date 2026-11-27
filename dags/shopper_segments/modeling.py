"""Building, training and comparing the segmentation models.

Each candidate is one scikit-learn Pipeline: log transform and scaling, then a
K-Means or Gaussian Mixture model. Because the preprocessing is saved inside the
Pipeline, new sessions always get the same treatment as the training data (the
original lab trained on scaled data but predicted on raw values).
"""

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from shopper_segments import settings


def candidate_grid(algorithms, k_min: int, k_max: int) -> list[dict]:
    """Every (algorithm, k) pair to train, e.g. [{"algorithm": "kmeans", "k": 4}, ...]."""
    algorithms = list(algorithms)
    unknown = sorted(set(algorithms) - set(settings.ALGORITHMS))
    if not algorithms or unknown:
        raise ValueError(f"algorithms must be a non-empty subset of {settings.ALGORITHMS}, got {algorithms}")
    if not 2 <= k_min <= k_max:
        raise ValueError(f"need 2 <= k_min <= k_max, got k_min={k_min}, k_max={k_max}")
    return [{"algorithm": a, "k": k}
            for a in settings.ALGORITHMS if a in algorithms
            for k in range(k_min, k_max + 1)]


def build_pipeline(algorithm: str, k: int) -> Pipeline:
    """Unfitted Pipeline: log1p on skewed columns, rates as they are, scaling, then the model."""
    preprocess = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), settings.LOG_FEATURES),
        ("rates", "passthrough", settings.RATE_FEATURES),
    ])
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=settings.RANDOM_STATE)
    elif algorithm == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                                random_state=settings.RANDOM_STATE)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return Pipeline([("preprocess", preprocess), ("scale", StandardScaler()), ("model", model)])


def evaluate(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    """Silhouette (higher is better), Davies-Bouldin (lower is better) and the smallest segment's share."""
    X = df[settings.FEATURES]
    Z = pipeline[:-1].transform(X)  # the scaled space the model clusters in
    labels = pipeline.predict(X)
    found = len(np.unique(labels))
    if found < 2:  # everything in one segment: silhouette is undefined, so score it as the worst
        return {"silhouette": -1.0, "davies_bouldin": None, "segments_found": found, "min_segment_share": 1.0}
    sample = settings.SILHOUETTE_SAMPLE if len(X) > settings.SILHOUETTE_SAMPLE else None
    silhouette = silhouette_score(Z, labels, sample_size=sample, random_state=settings.RANDOM_STATE)
    return {
        "silhouette": round(float(silhouette), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Z, labels)), 4),
        "segments_found": found,
        "min_segment_share": round(float(np.bincount(labels).min() / len(labels)), 4),
    }


def fit_candidate(train: pd.DataFrame, algorithm: str, k: int) -> tuple[Pipeline, dict]:
    """Train one candidate on the training sessions and score it."""
    start = time.perf_counter()
    pipeline = build_pipeline(algorithm, k).fit(train[settings.FEATURES])
    metrics = {"algorithm": algorithm, "k": k, **evaluate(pipeline, train)}
    metrics["fit_seconds"] = round(time.perf_counter() - start, 2)
    return pipeline, metrics


def pick_best(results: list[dict]) -> dict:
    """Highest silhouette wins. Ties go to the smaller k, then to K-Means (the simpler model)."""
    if not results:
        raise ValueError("no candidates to choose from")
    order = {name: i for i, name in enumerate(settings.ALGORITHMS)}
    return min(results, key=lambda r: (-r["silhouette"], r["k"], order[r["algorithm"]]))


def promotion_decision(best: dict, champion_silhouette, min_silhouette: float,
                       min_gain: float = 0.001) -> dict:
    """Should the best candidate replace the champion?

    It must reach min_silhouette and, if there is a champion, beat the champion's
    silhouette on the same training data by at least min_gain. A rerun on
    unchanged data therefore keeps the current champion.
    """
    score = best["silhouette"]
    if score < min_silhouette:
        return {"promote": False,
                "reason": f"silhouette {score:.4f} is below the minimum {min_silhouette:.4f}"}
    if champion_silhouette is None:
        return {"promote": True,
                "reason": f"silhouette {score:.4f} passes the minimum and there is no champion yet"}
    if score < champion_silhouette + min_gain:
        return {"promote": False,
                "reason": f"silhouette {score:.4f} does not beat the champion's "
                          f"{champion_silhouette:.4f} on the same data"}
    return {"promote": True,
            "reason": f"silhouette {score:.4f} beats the champion's {champion_silhouette:.4f}"}


def segment_profiles(pipeline: Pipeline, df: pd.DataFrame) -> list[dict]:
    """One row per segment: size, purchase rate and typical browsing behaviour."""
    labelled = df.assign(segment=pipeline.predict(df[settings.FEATURES]))
    grouped = labelled.groupby("segment")
    profiles = pd.DataFrame({
        "sessions": grouped.size(),
        "purchase_rate": grouped[settings.LABEL].mean(),
        "product_pages_median": grouped["ProductRelated"].median(),
        "product_minutes_median": grouped["ProductRelated_Duration"].median() / 60,
        "admin_pages_median": grouped["Administrative"].median(),
        "info_pages_median": grouped["Informational"].median(),
        "bounce_rate_median": grouped["BounceRates"].median(),
        "exit_rate_median": grouped["ExitRates"].median(),
        "page_value_mean": grouped["PageValues"].mean(),
        "new_visitor_share": grouped["VisitorType"].apply(lambda s: (s == "New_Visitor").mean()),
    })
    profiles["share"] = profiles["sessions"] / len(labelled)
    return [
        {key: int(value) if key in ("segment", "sessions") else round(float(value), 4)
         for key, value in row.items()}
        for row in profiles.reset_index().to_dict(orient="records")
    ]
