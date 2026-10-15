import json
import os
import tempfile
import joblib
from pathlib import Path
from typing import Any

from traffic_system.config import ensure_directories


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directories([path.parent])
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}. Run bootstrap/training first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(payload: Any, path: Path) -> None:
    ensure_directories([path.parent])
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(payload, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run bootstrap/training first.")
    try:
        return joblib.load(path)
    except (EOFError, ValueError) as exc:
        raise RuntimeError(
            f"Model bundle at {path} is corrupt or partially written. Retrain to regenerate."
        ) from exc
