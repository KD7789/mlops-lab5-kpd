"""
### Shopper segments: scoring

Scheduled on the `shopper_segment_champion` Asset: it runs by itself each time
the training DAG promotes a new champion, never on a clock.

1. **pick_model_version** reads the model version from the Asset event that
   triggered the run (or takes the current champion when the run is started by hand).
2. **load_new_sessions** keeps the Nov-Dec sessions, without duplicates.
3. **score_sessions** assigns every new session to a segment with that version.
4. **summarize_segments** compares the segment mix and purchase rates with the
   training months, and updates the `shopper_segment_scores` Asset.

Files go to `artifacts/scoring/<run_id>/`, and the latest summary is copied to
`artifacts/scoring/latest.json` for the dashboard.
"""

import logging
from datetime import datetime, timedelta, timezone

from airflow.sdk import Asset, dag, get_current_context, task

from shopper_segments import settings  # paths and constants only; heavy imports live in the tasks

log = logging.getLogger(__name__)

CHAMPION = Asset(name=settings.CHAMPION_ASSET_NAME, uri=settings.CHAMPION_ASSET_URI)
SCORES = Asset(name=settings.SCORES_ASSET_NAME, uri=settings.SCORES_ASSET_URI)


def scoring_folder():
    """This run's folder under artifacts/scoring/."""
    from shopper_segments import registry

    return registry.run_dir(settings.ARTIFACTS_DIR, get_current_context()["run_id"], kind="scoring")


def relative(path) -> str:
    return str(path.relative_to(settings.ARTIFACTS_DIR))


@task(retries=0)  # "no champion yet" won't fix itself, so don't retry
def pick_model_version() -> dict:
    """Use the version named in the triggering Asset event, or the current champion for a manual run."""
    from shopper_segments import registry

    context = get_current_context()
    events = [event for events in context["triggering_asset_events"].values() for event in events]
    events = [event for event in events if (event.extra or {}).get("version")]
    if events:
        event = events[-1]  # if several promotions queued up, score with the newest
        choice = {"version": event.extra["version"],
                  "trigger": f"Asset event from training run {getattr(event, 'source_run_id', None)}"}
    else:
        champion = registry.read_champion(settings.ARTIFACTS_DIR)
        if champion is None:
            raise RuntimeError("No champion yet. Run the training DAG and approve a model first.")
        choice = {"version": champion["version"], "trigger": "started by hand, using the current champion"}
    log.info("Scoring with %s (%s)", choice["version"], choice["trigger"])
    return choice


@task
def load_new_sessions() -> dict:
    """Keep the Nov-Dec sessions, without duplicates, and save them for this run."""
    from shopper_segments import data

    sessions = data.clean_sessions(data.load_sessions(settings.DATA_DIR / settings.SOURCE_FILE))
    new = data.split_by_month(sessions, settings.NEW_MONTHS)
    path = scoring_folder() / "new_sessions.csv"
    new.to_csv(path, index=False)
    log.info("%d new sessions from %s", len(new), ", ".join(settings.NEW_MONTHS))
    return {"sessions_file": relative(path), "rows": len(new)}


@task
def score_sessions(model: dict, new: dict) -> dict:
    """Assign every session to a segment. The saved Pipeline does its own preprocessing."""
    import pandas as pd

    from shopper_segments import registry, scoring

    pipeline, metadata = registry.load_version(settings.ARTIFACTS_DIR, model["version"])
    scored = scoring.score_sessions(pipeline, pd.read_csv(settings.ARTIFACTS_DIR / new["sessions_file"]))
    path = scoring_folder() / "scored_sessions.csv"
    scored.to_csv(path, index=False)
    log.info("Scored %d sessions with %s (%s, k=%d)", len(scored), metadata["version"],
             metadata["algorithm"], metadata["k"])
    return {"scored_file": relative(path)}


@task(outlets=[SCORES])
def summarize_segments(model: dict, scored: dict) -> dict:
    """Compare the new sessions with the training months and publish the summary."""
    import pandas as pd

    from shopper_segments import registry, scoring

    context = get_current_context()
    metadata = registry.version_metadata(settings.ARTIFACTS_DIR, model["version"])
    summary = scoring.summarize(pd.read_csv(settings.ARTIFACTS_DIR / scored["scored_file"]), metadata)
    summary.update({
        "run_id": context["run_id"],
        "trigger": model["trigger"],
        "algorithm": metadata["algorithm"],
        "k": metadata["k"],
        "scored_file": scored["scored_file"],
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    registry.write_json(scoring_folder() / "summary.json", summary)
    registry.write_json(settings.ARTIFACTS_DIR / "scoring" / "latest.json", summary)
    context["outlet_events"][SCORES].extra = {"champion_version": summary["champion_version"],
                                              "sessions": summary["sessions"]}
    line = "Segment %d: %.1f%% of new sessions (training %.1f%%), purchase rate %.1f%% (training %.1f%%)"
    for seg in summary["segments"]:
        log.info(line, seg["segment"], 100 * seg["share"], 100 * seg["training_share"],
                 100 * (seg["purchase_rate"] or 0), 100 * seg["training_purchase_rate"])
    return summary


@dag(
    dag_id="shopper_segments_scoring",
    dag_display_name="Shopper segments: scoring",
    description="Assign the Nov-Dec sessions to segments whenever a new champion is promoted",
    schedule=[CHAMPION],
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,  # active from the start, so the first promotion already triggers it
    default_args={
        "owner": "kenil",
        "retries": 2,
        "retry_delay": timedelta(seconds=10),
        "retry_exponential_backoff": 2.0,
        "max_retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=10),
    },
    tags=["mlops", "segmentation"],
    doc_md=__doc__,
)
def shopper_segments_scoring():
    model = pick_model_version()
    summarize_segments(model, score_sessions(model, load_new_sessions()))


shopper_segments_scoring()
