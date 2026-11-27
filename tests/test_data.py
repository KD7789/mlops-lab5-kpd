from shopper_segments import data, settings


def failed_checks(report):
    return {c["name"] for c in report["checks"] if not c["passed"]}


def test_real_file_passes_validation(sessions):
    report = data.validate_sessions(sessions)
    assert report["passed"], failed_checks(report)
    assert report["rows"] == 12330
    assert report["duplicate_rows"] == 125


def test_missing_column_fails(sessions):
    report = data.validate_sessions(sessions.drop(columns=["PageValues"]))
    assert not report["passed"]
    assert "columns" in failed_checks(report)


def test_bad_values_fail(sessions):
    bad = sessions.copy()
    bad.loc[0, "ProductRelated_Duration"] = -5
    bad.loc[1, "BounceRates"] = 1.5
    bad.loc[2, "Month"] = "Smarch"
    report = data.validate_sessions(bad)
    assert not report["passed"]
    assert failed_checks(report) == {"non_negative", "rates_in_range", "known_months"}


def test_too_few_rows_fails(sessions):
    report = data.validate_sessions(sessions.head(500))
    assert failed_checks(report) == {"row_count"}


def test_needs_training_months(sessions):
    report = data.validate_sessions(data.split_by_month(sessions, settings.NEW_MONTHS))
    assert failed_checks(report) == {"enough_training_rows"}


def test_clean_and_split(sessions):
    clean = data.clean_sessions(sessions)
    assert len(clean) == 12330 - 125
    train = data.split_by_month(clean, settings.TRAIN_MONTHS)
    new = data.split_by_month(clean, settings.NEW_MONTHS)
    assert (len(train), len(new)) == (7517, 4688)
    assert set(train["Month"]).isdisjoint(new["Month"])
