from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_pipeline import bootstrap_project


def bootstrap() -> None:
    results = bootstrap_project()
    print(f"Raw rows: {results['raw_rows']}")
    print(f"Processed rows: {results['processed_rows']}")
    print(f"Model bundle: {results['model_bundle']}")
    print(f"Forecast file: {results['forecast_file']}")
    print(f"Spark pipeline: {results['spark_pipeline']}")


if __name__ == "__main__":
    bootstrap()
