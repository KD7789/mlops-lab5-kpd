# Airflow 3.4.0 plus this lab's pinned Python packages (requirements.txt).
# Pinning apache-airflow to the image's own version stops pip from upgrading
# or downgrading Airflow while it installs the requirements.
FROM apache/airflow:3.4.0
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
