"""Structured JSONL telemetry logger for runs and episodes."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_json_value(value: Any) -> Any:
    """Convert nested runtime values into JSON-safe content."""
    if is_dataclass(value):
        value = asdict(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]

    return repr(value)


@dataclass(frozen=True)
class TelemetryLoggerConfig:
    """Configuration for per-run telemetry output."""

    logs_root: Path = Path("logs")
    run_name: str | None = None
    session_id: str | None = None
    events_filename: str = "events.jsonl"


@dataclass
class JsonlTelemetryLogger:
    """Append-only JSONL telemetry writer with session/run metadata."""

    config: TelemetryLoggerConfig = field(default_factory=TelemetryLoggerConfig)
    _session_id: str = field(init=False)
    _run_id: str = field(init=False)
    _run_dir: Path = field(init=False)
    _events_path: Path = field(init=False)
    _episode_next_step_index: dict[str, int] = field(default_factory=dict, init=False)
    _failed_write_count: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _current_episode_index: int = field(default=0, init=False)
    _handle: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        session_id = self.config.session_id or uuid.uuid4().hex
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_suffix = (self.config.run_name or "run").strip().replace(" ", "-")
        run_id = f"{run_stamp}_{run_suffix}_{session_id[:8]}"

        run_dir = self.config.logs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / self.config.events_filename

        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_run_dir", run_dir)
        object.__setattr__(self, "_events_path", events_path)
        object.__setattr__(self, "_handle", events_path.open("a", encoding="ascii"))

    @property
    def session_id(self) -> str:
        """Stable session identifier for this logger instance."""
        return self._session_id

    @property
    def run_id(self) -> str:
        """Unique run identifier used for output namespacing."""
        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Directory containing this run's telemetry files."""
        return self._run_dir

    @property
    def events_path(self) -> Path:
        """Path to JSONL events file."""
        return self._events_path

    @property
    def failed_write_count(self) -> int:
        """Count of write failures swallowed to protect control loop."""
        return self._failed_write_count

    def close(self) -> None:
        """Flush and close the active JSONL file handle."""
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.flush()
            self._handle.close()
        except Exception:  # pragma: no cover - defensive close path
            LOGGER.exception("Failed closing telemetry logger at %s", self._events_path)

    def __enter__(self) -> JsonlTelemetryLogger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start_episode(
        self,
        episode_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Start a new episode and emit episode_start event."""
        if episode_id is None:
            self._current_episode_index += 1
            episode_id = f"episode-{self._current_episode_index:05d}"

        self._episode_next_step_index[episode_id] = 0
        self.log_event(
            event_type="episode_start",
            episode_id=episode_id,
            payload={"metadata": metadata or {}},
        )
        return episode_id

    def log_step(
        self,
        *,
        episode_id: str,
        action: Any,
        pre_state: Any,
        post_state: Any,
        reward: float,
        done: bool,
        info: Mapping[str, Any] | None = None,
        step_index: int | None = None,
    ) -> int:
        """Log one environment step for an episode."""
        resolved_step_index = self._resolve_step_index(episode_id=episode_id, step_index=step_index)
        self.log_event(
            event_type="step",
            episode_id=episode_id,
            step_index=resolved_step_index,
            payload={
                "action": action,
                "pre_state": pre_state,
                "post_state": post_state,
                "reward": reward,
                "done": done,
                "info": info or {},
            },
        )
        return resolved_step_index

    def log_terminal(
        self,
        *,
        episode_id: str,
        reason: str,
        terminal_state: Any | None = None,
        info: Mapping[str, Any] | None = None,
        step_index: int | None = None,
    ) -> int:
        """Log terminal event for an episode."""
        resolved_step_index = self._resolve_step_index(episode_id=episode_id, step_index=step_index)
        self.log_event(
            event_type="terminal",
            episode_id=episode_id,
            step_index=resolved_step_index,
            payload={
                "reason": reason,
                "terminal_state": terminal_state,
                "info": info or {},
            },
        )
        return resolved_step_index

    def log_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        episode_id: str | None = None,
        step_index: int | None = None,
        timestamp_utc: str | None = None,
    ) -> None:
        """Write one telemetry event, swallowing failures by design."""
        if self._closed:
            LOGGER.warning("Ignoring telemetry event after close: event_type=%s", event_type)
            return

        event: dict[str, Any] = {
            "event_type": event_type,
            "timestamp_utc": timestamp_utc or _now_iso_utc(),
            "session_id": self._session_id,
            "run_id": self._run_id,
        }
        if episode_id is not None:
            event["episode_id"] = episode_id
        if step_index is not None:
            event["step_index"] = int(step_index)
        if payload is not None:
            for key, value in payload.items():
                event[key] = _safe_json_value(value)

        self._write_event(event)

    def _resolve_step_index(self, *, episode_id: str, step_index: int | None) -> int:
        if step_index is not None:
            next_index = int(step_index) + 1
            previous = self._episode_next_step_index.get(episode_id, 0)
            self._episode_next_step_index[episode_id] = max(previous, next_index)
            return int(step_index)

        next_index = self._episode_next_step_index.get(episode_id, 0)
        self._episode_next_step_index[episode_id] = next_index + 1
        return next_index

    def _write_event(self, event: Mapping[str, Any]) -> None:
        try:
            serialized = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
            self._handle.write(serialized + "\n")
            self._handle.flush()
        except Exception:
            self._failed_write_count += 1
            LOGGER.exception(
                "Telemetry write failure (count=%s) to %s",
                self._failed_write_count,
                self._events_path,
            )


def load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    """Load and parse JSONL events from disk."""
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        if isinstance(raw, dict):
            events.append(raw)
    return events
