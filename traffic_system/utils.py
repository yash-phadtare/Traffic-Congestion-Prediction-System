import json
import joblib
from pathlib import Path
from typing import Any

from traffic_system.config import ensure_directories

def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directories([path.parent])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_joblib(payload: Any, path: Path) -> None:
    ensure_directories([path.parent])
    joblib.dump(payload, path)

def load_joblib(path: Path) -> Any:
    return joblib.load(path)
