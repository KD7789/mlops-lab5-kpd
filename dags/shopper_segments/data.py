"""Loading, validating and splitting the sessions file."""

import pandas as pd

from shopper_segments import settings


def load_sessions(path) -> pd.DataFrame:
    """Read the sessions CSV. pandas parses the TRUE/FALSE columns as booleans."""
    return pd.read_csv(path)


def _check(name: str, passed, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_sessions(df: pd.DataFrame) -> dict:
    """Run the data checks and return a JSON-friendly report.

    The report's "passed" is True only if every check passed. Duplicate rows are
    counted but don't fail validation, because cleaning removes them.
    """
    missing = [c for c in settings.EXPECTED_COLUMNS if c not in df.columns]
    checks = [
        _check("columns", not missing,
               f"missing columns: {missing}" if missing else "all 18 expected columns present"),
        _check("row_count", len(df) >= settings.MIN_ROWS,
               f"{len(df)} rows (minimum {settings.MIN_ROWS})"),
    ]

    if not missing:  # the remaining checks need the columns to exist
        used = settings.FEATURES + ["Month", "VisitorType", settings.LABEL]
        nulls = int(df[used].isna().sum().sum())
        numeric = df[settings.FEATURES].apply(pd.to_numeric, errors="coerce")
        non_numeric = int((numeric.isna() & df[settings.FEATURES].notna()).sum().sum())
        negative = int((numeric[settings.LOG_FEATURES] < 0).sum().sum())
        rates = numeric[settings.RATE_FEATURES]
        out_of_range = int(((rates < 0) | (rates > 1)).sum().sum())
        bad_months = sorted(set(df["Month"].dropna().astype(str)) - set(settings.ALL_MONTHS))
        bad_visitors = sorted(set(df["VisitorType"].dropna().astype(str)) - set(settings.VISITOR_TYPES))
        label_ok = df[settings.LABEL].dropna().isin([True, False]).all()
        train_rows = int(df["Month"].isin(settings.TRAIN_MONTHS).sum())
        checks += [
            _check("no_missing_values", nulls == 0, f"{nulls} missing values in used columns"),
            _check("numeric_features", non_numeric == 0, f"{non_numeric} non-numeric feature values"),
            _check("non_negative", negative == 0, f"{negative} negative counts, durations or page values"),
            _check("rates_in_range", out_of_range == 0, f"{out_of_range} bounce/exit rates outside [0, 1]"),
            _check("known_months", not bad_months,
                   f"unknown months: {bad_months}" if bad_months else "all months known"),
            _check("known_visitor_types", not bad_visitors,
                   f"unknown visitor types: {bad_visitors}" if bad_visitors else "all visitor types known"),
            _check("boolean_label", label_ok,
                   "Revenue is TRUE/FALSE" if label_ok else "Revenue has non-boolean values"),
            _check("enough_training_rows", train_rows >= settings.MIN_TRAIN_ROWS,
                   f"{train_rows} rows in training months (minimum {settings.MIN_TRAIN_ROWS})"),
        ]

    return {
        "passed": all(c["passed"] for c in checks),
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
        "checks": checks,
    }


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def split_by_month(df: pd.DataFrame, months) -> pd.DataFrame:
    """Return the sessions from the given months."""
    return df[df["Month"].isin(months)].reset_index(drop=True)
