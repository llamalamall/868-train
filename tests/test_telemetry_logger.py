"""Tests for JSONL telemetry logging and episode utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.telemetry.logger import JsonlTelemetryLogger, TelemetryLoggerConfig, load_jsonl_events
from src.telemetry.metrics import load_and_summarize, load_episode_replay


@dataclass
class _Unserializable:
    """Simple object to verify safe serialization fallback."""

    name: str


def test_logger_writes_namespaced_events_and_replayable_episode(tmp_path: Path) -> None:
    logger = JsonlTelemetryLogger(
        TelemetryLoggerConfig(logs_root=tmp_path, run_name="task-10", session_id="session-fixed")
    )
    episode_id = logger.start_episode("episode-00001", metadata={"difficulty": "normal"})
    step0 = logger.log_step(
        episode_id=episode_id,
        action="move_up",
        pre_state={"health": 10},
        post_state={"health": 10},
        reward=0.5,
        done=False,
        info={"input_ok": True},
    )
    step1 = logger.log_step(
        episode_id=episode_id,
        action="wait",
        pre_state={"health": 10},
        post_state={"health": 9},
        reward=-1.0,
        done=True,
        info={"damage": 1},
    )
    terminal_step = logger.log_terminal(
        episode_id=episode_id,
        reason="fail_state",
        terminal_state={"health": -1},
        info={"source": "memory"},
    )
    logger.close()

    assert step0 == 0
    assert step1 == 1
    assert terminal_step == 2
    assert logger.run_dir.parent == tmp_path
    assert logger.events_path.exists()

    events = load_jsonl_events(logger.events_path)
    assert [event["event_type"] for event in events] == [
        "episode_start",
        "step",
        "step",
        "terminal",
    ]
    assert {event["session_id"] for event in events} == {"session-fixed"}
    assert {event["run_id"] for event in events} == {logger.run_id}

    replay = load_episode_replay(logger.events_path, "episode-00001")
    assert len(replay.step_events) == 2
    assert replay.step_events[0]["action"] == "move_up"
    assert replay.step_events[1]["action"] == "wait"
    assert replay.terminal_event is not None
    assert replay.terminal_event["reason"] == "fail_state"


def test_logger_auto_step_index_isolated_per_episode(tmp_path: Path) -> None:
    logger = JsonlTelemetryLogger(TelemetryLoggerConfig(logs_root=tmp_path))
    ep1 = logger.start_episode("episode-A")
    ep2 = logger.start_episode("episode-B")

    ep1_step0 = logger.log_step(
        episode_id=ep1,
        action="a",
        pre_state={},
        post_state={},
        reward=0.0,
        done=False,
    )
    ep1_step1 = logger.log_step(
        episode_id=ep1,
        action="b",
        pre_state={},
        post_state={},
        reward=0.0,
        done=False,
    )
    ep2_step0 = logger.log_step(
        episode_id=ep2,
        action="c",
        pre_state={},
        post_state={},
        reward=0.0,
        done=False,
    )
    logger.close()

    assert ep1_step0 == 0
    assert ep1_step1 == 1
    assert ep2_step0 == 0


def test_logger_write_failures_do_not_raise(tmp_path: Path) -> None:
    class _FailingHandle:
        def write(self, _: str) -> int:
            raise OSError("simulated write failure")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    logger = JsonlTelemetryLogger(TelemetryLoggerConfig(logs_root=tmp_path))
    logger._handle = _FailingHandle()
    logger.log_event(
        event_type="step",
        episode_id="episode-x",
        step_index=0,
        payload={"state": _Unserializable(name="opaque")},
    )
    logger.close()

    assert logger.failed_write_count == 1


def test_load_and_summarize_returns_per_episode_metrics(tmp_path: Path) -> None:
    logger = JsonlTelemetryLogger(TelemetryLoggerConfig(logs_root=tmp_path))
    ep1 = logger.start_episode("episode-00002")
    logger.log_step(
        episode_id=ep1,
        action="left",
        pre_state={},
        post_state={},
        reward=1.25,
        done=False,
    )
    logger.log_step(
        episode_id=ep1,
        action="right",
        pre_state={},
        post_state={},
        reward=0.75,
        done=True,
    )
    logger.log_terminal(episode_id=ep1, reason="completed")
    logger.close()

    summaries = load_and_summarize(logger.events_path)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.episode_id == "episode-00002"
    assert summary.step_count == 2
    assert summary.total_reward == 2.0
    assert summary.done_seen is True
    assert summary.terminal_reason == "completed"
