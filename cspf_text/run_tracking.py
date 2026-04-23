from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class RunTracker:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "run.log"
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.state_path = self.output_dir / "run_state.json"
        self.state: dict[str, Any] = {
            "status": "initialized",
            "current_stage": "initialized",
            "started_at": _utc_timestamp(),
            "updated_at": _utc_timestamp(),
            "completed_stages": [],
            "artifacts": {},
        }
        self._write_json(self.state_path, self.state)

    def _write_json(self, path: Path, payload: Any) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
        temp_path.replace(path)

    def _append_text(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log(self, message: str, *, stage: str | None = None, extra: dict[str, Any] | None = None) -> None:
        timestamp = _utc_timestamp()
        prefix = f"[{timestamp}]"
        if stage:
            prefix += f" [{stage}]"
        line = f"{prefix} {message}"
        print(line, flush=True)
        self._append_text(self.log_path, line)

        event = {"timestamp": timestamp, "message": message}
        if stage:
            event["stage"] = stage
        if extra:
            event["extra"] = extra
        self._append_text(self.jsonl_path, json.dumps(event, ensure_ascii=False, default=_json_default))

    def set_stage(self, stage: str, *, status: str = "running", extra: dict[str, Any] | None = None) -> None:
        self.state["status"] = status
        self.state["current_stage"] = stage
        self.state["updated_at"] = _utc_timestamp()
        if extra:
            self.state.update(extra)
        self._write_json(self.state_path, self.state)

    def complete_stage(self, stage: str, *, extra: dict[str, Any] | None = None) -> None:
        completed = self.state.setdefault("completed_stages", [])
        if stage not in completed:
            completed.append(stage)
        self.state["current_stage"] = stage
        self.state["updated_at"] = _utc_timestamp()
        if extra:
            self.state.update(extra)
        self._write_json(self.state_path, self.state)

    def save_json(self, filename: str, payload: Any, *, artifact_key: str | None = None) -> Path:
        path = self.output_dir / filename
        self._write_json(path, payload)
        if artifact_key:
            self.state.setdefault("artifacts", {})[artifact_key] = str(path)
            self.state["updated_at"] = _utc_timestamp()
            self._write_json(self.state_path, self.state)
        return path

    def save_pickle(self, filename: str, payload: Any, *, artifact_key: str | None = None) -> Path:
        path = self.output_dir / filename
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        if artifact_key:
            self.state.setdefault("artifacts", {})[artifact_key] = str(path)
            self.state["updated_at"] = _utc_timestamp()
            self._write_json(self.state_path, self.state)
        return path

    def finalize(self, status: str, *, extra: dict[str, Any] | None = None) -> None:
        self.state["status"] = status
        self.state["updated_at"] = _utc_timestamp()
        if status == "completed":
            self.state["completed_at"] = _utc_timestamp()
        if extra:
            self.state.update(extra)
        self._write_json(self.state_path, self.state)
