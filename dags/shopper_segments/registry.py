"""Where each run's files go, and which model is the current champion.

Layout under the artifacts folder:

    runs/<run>/                       one folder per training run
        validation.json, train.csv, selection.json, summary.json
        candidates/<algorithm>_k<k>.joblib
    models/history/v1/, v2/, ...      every promoted model: model.joblib + metadata.json
    models/champion.json              the current champion (a copy of its metadata)
    scoring/<run>/                    one folder per scoring run
        scored_sessions.csv, summary.json
    scoring/latest.json               the most recent scoring summary

Paths stored inside JSON files are relative to the artifacts folder, so the
dashboard container can read them from a different mount point.
"""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib


def safe_name(run_id: str) -> str:
    """Turn an Airflow run_id (it contains ':' and '+') into a folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def run_dir(artifacts_dir, run_id: str, kind: str = "runs") -> Path:
    path = Path(artifacts_dir) / kind / safe_name(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj) -> None:
    """Write JSON atomically (temp file + rename), so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_model(pipeline, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path):
    return joblib.load(path)


def champion_file(artifacts_dir) -> Path:
    return Path(artifacts_dir) / "models" / "champion.json"


def read_champion(artifacts_dir):
    """The current champion's metadata, or None if nothing has been promoted yet."""
    return read_json(champion_file(artifacts_dir))


def load_champion_model(artifacts_dir):
    champion = read_champion(artifacts_dir)
    if champion is None:
        raise FileNotFoundError("no champion model has been promoted yet")
    return load_model(Path(artifacts_dir) / champion["model_file"]), champion


def _next_version(history_dir: Path) -> str:
    numbers = [int(m.group(1)) for p in history_dir.glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name))]
    return f"v{max(numbers, default=0) + 1}"


def promote(artifacts_dir, model_path, metadata: dict) -> dict:
    """Copy a candidate into models/history/vN/ and make it the champion."""
    artifacts_dir = Path(artifacts_dir)
    history_dir = artifacts_dir / "models" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(history_dir)
    target = history_dir / version
    target.mkdir()
    shutil.copy2(model_path, target / "model.joblib")
    record = {
        **metadata,
        "version": version,
        "model_file": str((target / "model.joblib").relative_to(artifacts_dir)),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(target / "metadata.json", record)
    write_json(champion_file(artifacts_dir), record)
    return record
