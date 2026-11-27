"""Paths, column lists and defaults shared by the DAGs, the tasks and the tests."""

from pathlib import Path

# Folders inside the Airflow containers (mounted from the repo in docker-compose.yaml).
DATA_DIR = Path("/opt/airflow/data")
ARTIFACTS_DIR = Path("/opt/airflow/artifacts")
SOURCE_FILE = "online_shoppers_intention.csv"

# The 18 columns of the UCI file, in file order.
EXPECTED_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend", "Revenue",
]

# The 9 browsing-behaviour columns the model clusters on. Counts, durations and
# page values are heavily skewed, so they get a log transform; rates are already in [0, 1].
LOG_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "PageValues",
]
RATE_FEATURES = ["BounceRates", "ExitRates"]
FEATURES = LOG_FEATURES + RATE_FEATURES

# Whether the session ended in a purchase. Only used to describe segments, never to train.
LABEL = "Revenue"

ALL_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TRAIN_MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct"]
NEW_MONTHS = ["Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

# Validation thresholds.
MIN_ROWS = 1000
MIN_TRAIN_ROWS = 500

# Model search.
ALGORITHMS = ["kmeans", "gmm"]
RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 10_000  # silhouette is O(n^2), so larger datasets are sampled
