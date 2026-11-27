"""Assigning new sessions to segments and comparing them with the training data."""

import pandas as pd

from shopper_segments import settings


def score_sessions(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Add a "segment" column. The Pipeline does its own preprocessing on the raw columns."""
    return df.assign(segment=pipeline.predict(df[settings.FEATURES]))


def summarize(scored: pd.DataFrame, champion: dict) -> dict:
    """Segment mix and purchase rates of the new sessions, next to the training profile."""
    training = {p["segment"]: p for p in champion["profiles"]}
    grouped = scored.groupby("segment")
    sizes = grouped.size()
    purchase = grouped[settings.LABEL].mean()
    segments = []
    for segment in sorted(training):
        sessions = int(sizes.get(segment, 0))
        segments.append({
            "segment": int(segment),
            "sessions": sessions,
            "share": round(sessions / len(scored), 4),
            "purchase_rate": round(float(purchase[segment]), 4) if sessions else None,
            "training_share": training[segment]["share"],
            "training_purchase_rate": training[segment]["purchase_rate"],
        })
    return {
        "champion_version": champion["version"],
        "sessions": int(len(scored)),
        "purchase_rate": round(float(scored[settings.LABEL].mean()), 4),
        "months": sorted(scored["Month"].unique().tolist(), key=settings.ALL_MONTHS.index),
        "segments": segments,
    }
