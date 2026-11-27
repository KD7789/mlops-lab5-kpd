"""
### Shopper segments: training

Runs weekly (or on demand from the trigger form) and keeps the champion
segmentation model up to date.

1. **validate_data** runs 10 checks on the sessions file. **check_data** branches:
   bad data goes to **quarantine_batch**, which fails the run on purpose so it
   gets noticed. Nothing downstream runs.
2. **prepare_training_data** removes duplicates and keeps the Feb-Oct sessions.
3. **model_search** trains one candidate per (algorithm, k) in parallel, using
   dynamic task mapping, then picks the highest silhouette.
4. **promotion** compares the winner with the current champion on the same data.
   If it is good enough, a person approves it in the UI (Required Actions) and
   **promote_model** makes it the champion. That update of the champion Asset
   starts the scoring DAG.
5. **write_run_summary** records what happened, whichever path the run took.

Files go to `artifacts/runs/<run_id>/`. Only small metadata travels in XCom.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from airflow.providers.standard.operators.hitl import ApprovalOperator
from airflow.sdk import Asset, Param, dag, get_current_context, task, task_group

from shopper_segments import data, modeling, registry, settings

log = logging.getLogger(__name__)

CHAMPION = Asset(name=settings.CHAMPION_ASSET_NAME, uri=settings.CHAMPION_ASSET_URI)


def run_folder():
    """This run's folder under artifacts/runs/."""
    return registry.run_dir(settings.ARTIFACTS_DIR, get_current_context()["run_id"])


def relative(path) -> str:
    return str(path.relative_to(settings.ARTIFACTS_DIR))


@task
def validate_data() -> dict:
    """Run the data checks and save the report as validation.json."""
    source = settings.DATA_DIR / get_current_context()["params"]["source_file"]
    report = data.validate_sessions(data.load_sessions(source))
    report["source_file"] = source.name
    registry.write_json(run_folder() / "validation.json", report)
    for check in report["checks"]:
        log.info("%s %s: %s", "PASS" if check["passed"] else "FAIL", check["name"], check["detail"])
    return report


@task.branch
def check_data(report: dict) -> str:
    """Continue to training only if every check passed."""
    return "prepare_training_data" if report["passed"] else "quarantine_batch"


@task(retries=0)
def quarantine_batch(report: dict) -> None:
    """Stop here and fail the run, so bad data can't reach a model."""
    failed = [f"{c['name']} ({c['detail']})" for c in report["checks"] if not c["passed"]]
    raise RuntimeError(f"{report['source_file']} failed validation: {'; '.join(failed)}. "
                       "Details are in validation.json in this run's folder.")


@task
def prepare_training_data() -> dict:
    """Drop duplicate rows, keep the training months and save them as train.csv."""
    params = get_current_context()["params"]
    sessions = data.clean_sessions(data.load_sessions(settings.DATA_DIR / params["source_file"]))
    train = data.split_by_month(sessions, settings.TRAIN_MONTHS)
    path = run_folder() / "train.csv"
    train.to_csv(path, index=False)
    log.info("Training on %d sessions from %s", len(train), ", ".join(settings.TRAIN_MONTHS))
    return {"train_file": relative(path), "rows": len(train)}


@task
def plan_candidates(prepared: dict) -> list[dict]:
    """Turn the trigger-form choices into one (algorithm, k) spec per candidate."""
    params = get_current_context()["params"]
    specs = modeling.candidate_grid(params["algorithms"], params["k_min"], params["k_max"])
    log.info("%d candidates to train on %d sessions", len(specs), prepared["rows"])
    return specs


@task(map_index_template="{{ candidate_label }}", max_active_tis_per_dagrun=3)
def train_candidate(spec: dict, prepared: dict) -> dict:
    """Train and score one candidate. Each one is a separate mapped task instance."""
    context = get_current_context()
    context["candidate_label"] = f"{spec['algorithm']} k={spec['k']}"  # name shown in the UI
    train = pd.read_csv(settings.ARTIFACTS_DIR / prepared["train_file"])
    pipeline, metrics = modeling.fit_candidate(train, spec["algorithm"], spec["k"])
    path = run_folder() / "candidates" / f"{spec['algorithm']}_k{spec['k']}.joblib"
    registry.save_model(pipeline, path)
    log.info("%s", metrics)
    return {**metrics, "model_file": relative(path)}


@task
def select_best(results, prepared: dict) -> dict:
    """Pick the highest silhouette, describe its segments and save selection.json."""
    results = sorted(results, key=lambda r: -r["silhouette"])
    best = dict(modeling.pick_best(results))
    train = pd.read_csv(settings.ARTIFACTS_DIR / prepared["train_file"])
    pipeline = registry.load_model(settings.ARTIFACTS_DIR / best["model_file"])
    best["profiles"] = modeling.segment_profiles(pipeline, train)
    registry.write_json(run_folder() / "selection.json", {"best": best, "candidates": results})
    log.info("Best: %s k=%d, silhouette %.4f", best["algorithm"], best["k"], best["silhouette"])
    return best


def review_markdown(best: dict, decision: dict, min_silhouette: float) -> str:
    """What the reviewer sees in the approval form."""
    champion = (f"current champion {decision['champion_version']} scores "
                f"{decision['champion_silhouette']:.4f} on the same data"
                if decision["champion_version"] else "no champion yet")
    rows = "\n".join(
        f"| {p['segment']} | {p['share']:.1%} | {p['purchase_rate']:.1%} | "
        f"{p['product_pages_median']:.0f} | {p['product_minutes_median']:.1f} | {p['page_value_mean']:.1f} |"
        for p in best["profiles"]
    )
    return (
        f"**Candidate:** {best['algorithm']} with k = {best['k']}, silhouette {best['silhouette']:.4f} "
        f"(minimum {min_silhouette:.2f}); {champion}.\n\n"
        f"**Gate:** {decision['reason']}.\n\n"
        "| Segment | Share of sessions | Purchase rate | Product pages (median) "
        "| Minutes on product pages (median) | Page value (mean) |\n"
        "|---|---|---|---|---|---|\n"
        f"{rows}\n\n"
        "Approve to make this the champion. Promoting it starts the scoring DAG on the Nov-Dec sessions."
    )


@task.branch
def check_gate(best: dict, prepared: dict) -> str:
    """Ask for approval only if the winner passes the minimum and beats the champion."""
    context = get_current_context()
    min_silhouette = context["params"]["min_silhouette"]
    champion = registry.read_champion(settings.ARTIFACTS_DIR)
    champion_score = None
    if champion:  # re-score the champion on this run's training data, so the comparison is fair
        train = pd.read_csv(settings.ARTIFACTS_DIR / prepared["train_file"])
        model = registry.load_model(settings.ARTIFACTS_DIR / champion["model_file"])
        champion_score = modeling.evaluate(model, train)["silhouette"]
    decision = modeling.promotion_decision(best, champion_score, min_silhouette)
    decision["champion_version"] = champion["version"] if champion else None
    decision["champion_silhouette"] = champion_score
    registry.write_json(run_folder() / "gate.json", decision)
    context["ti"].xcom_push(key="decision", value=decision)
    context["ti"].xcom_push(key="review", value=review_markdown(best, decision, min_silhouette))
    log.info("Gate: %s", decision["reason"])
    return "promotion.approve_model" if decision["promote"] else "promotion.keep_champion"


@task(outlets=[CHAMPION])
def promote_model(best: dict, prepared: dict) -> dict:
    """Make the approved candidate the champion. Updating the Asset starts the scoring DAG."""
    context = get_current_context()
    ti = context["ti"]
    decision = ti.xcom_pull(task_ids="promotion.check_gate", key="decision")
    approval = ti.xcom_pull(task_ids="promotion.approve_model") or {}
    metadata = {
        **{key: best[key] for key in ("algorithm", "k", "silhouette", "davies_bouldin", "min_segment_share")},
        "profiles": best["profiles"],
        "train_rows": prepared["rows"],
        "train_months": settings.TRAIN_MONTHS,
        "run_id": context["run_id"],
        "gate_reason": decision["reason"],
        "approved_by": approval.get("responded_by_user"),
    }
    record = registry.promote(settings.ARTIFACTS_DIR, settings.ARTIFACTS_DIR / best["model_file"], metadata)
    context["outlet_events"][CHAMPION].extra = {
        key: record[key] for key in ("version", "algorithm", "k", "silhouette")
    }
    log.info("Promoted %s (%s k=%d) to champion", record["version"], record["algorithm"], record["k"])
    return {"version": record["version"], "model_file": record["model_file"]}


@task
def keep_champion() -> None:
    """The winner didn't pass the gate, so the current champion stays."""
    decision = get_current_context()["ti"].xcom_pull(task_ids="promotion.check_gate", key="decision")
    log.info("Keeping the current champion: %s", decision["reason"])


@task(trigger_rule="none_failed")
def write_run_summary() -> dict:
    """Record the outcome in summary.json. Runs after either promotion path, or when both were skipped."""
    context = get_current_context()
    ti = context["ti"]
    report = ti.xcom_pull(task_ids="validate_data")
    best = ti.xcom_pull(task_ids="model_search.select_best")
    decision = ti.xcom_pull(task_ids="promotion.check_gate", key="decision")
    approval = ti.xcom_pull(task_ids="promotion.approve_model")
    promoted = ti.xcom_pull(task_ids="promotion.promote_model")
    if report and not report["passed"]:
        outcome = "invalid_data"
    elif promoted:
        outcome = "promoted"
    elif decision and decision["promote"]:
        outcome = "rejected_by_reviewer"
    else:
        outcome = "kept_champion"
    summary = {
        "run_id": context["run_id"],
        "outcome": outcome,
        "params": dict(context["params"]),
        "rows_checked": report["rows"] if report else None,
        "best": {k: v for k, v in best.items() if k != "profiles"} if best else None,
        "gate": decision,
        "review": approval["chosen_options"][0] if approval else None,
        "champion_version": promoted["version"] if promoted else (decision or {}).get("champion_version"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    registry.write_json(run_folder() / "summary.json", summary)
    log.info("Run outcome: %s", outcome)
    return summary


@dag(
    dag_id="shopper_segments_training",
    dag_display_name="Shopper segments: training",
    description="Validate sessions, train K-Means and Gaussian Mixture candidates in parallel, "
                "and promote the best one after approval",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,  # two runs promoting at once could overwrite each other's champion
    default_args={
        "owner": "kenil",
        "retries": 2,
        "retry_delay": timedelta(seconds=10),
        "retry_exponential_backoff": 2.0,  # 10 s, then 20 s
        "max_retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(minutes=10),
    },
    params={
        "k_min": Param(4, type="integer", minimum=2, maximum=12, title="Smallest k",
                       description="Fewest segments to try. Below 4 the segments are too coarse to act on."),
        "k_max": Param(8, type="integer", minimum=2, maximum=12, title="Largest k",
                       description="Most segments to try."),
        "algorithms": Param(["kmeans", "gmm"], type="array", title="Algorithms",
                            items={"type": "string", "enum": settings.ALGORITHMS},
                            description="K-Means, Gaussian Mixture, or both."),
        "min_silhouette": Param(0.25, type="number", minimum=0, maximum=1, title="Minimum silhouette",
                                description="The winner needs at least this silhouette to be considered."),
        "source_file": Param(settings.SOURCE_FILE, type="string", pattern=r"^[A-Za-z0-9_.-]+\.csv$",
                             title="Sessions file", description="A CSV file in the repo's data/ folder."),
    },
    tags=["mlops", "segmentation"],
    doc_md=__doc__,
)
def shopper_segments_training():
    report = validate_data()
    branch = check_data(report)
    prepared = prepare_training_data()
    branch >> [prepared, quarantine_batch(report)]

    @task_group(group_id="model_search")
    def model_search():
        specs = plan_candidates(prepared)
        results = train_candidate.partial(prepared=prepared).expand(spec=specs)
        return select_best(results, prepared)

    @task_group(group_id="promotion")
    def promotion(best):
        gate = check_gate(best, prepared)
        approve = ApprovalOperator(
            task_id="approve_model",
            subject="Promote the new shopper segment model?",
            body="{{ ti.xcom_pull(task_ids='promotion.check_gate', key='review') }}",
            defaults="Reject",  # no answer within the timeout means no promotion
            response_timeout=timedelta(hours=24),
            retries=0,
        )
        gate >> [approve, keep_champion()]
        approve >> promote_model(best, prepared)

    # The summary waits for both ends of the promotion group (promote_model and keep_champion).
    promotion(model_search()) >> write_run_summary()


shopper_segments_training()
