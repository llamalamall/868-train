"""Telemetry loading, replay, and summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.telemetry.logger import load_jsonl_events


@dataclass(frozen=True)
class EpisodeReplay:
    """Single-episode replay reconstructed from logged events."""

    episode_id: str
    events: tuple[dict[str, Any], ...]
    step_events: tuple[dict[str, Any], ...]
    terminal_event: dict[str, Any] | None


@dataclass(frozen=True)
class EpisodeSummary:
    """Aggregate stats for one episode."""

    episode_id: str
    step_count: int
    total_reward: float
    done_seen: bool
    terminal_reason: str | None
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


def group_events_by_episode(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group events by episode identifier, excluding events without episode_id."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        episode_id = event.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        grouped.setdefault(episode_id, []).append(event)
    return grouped


def replay_episode(events: Iterable[dict[str, Any]], episode_id: str) -> EpisodeReplay:
    """Reconstruct one episode event stream and sorted steps."""
    selected = [
        event
        for event in events
        if isinstance(event.get("episode_id"), str) and event.get("episode_id") == episode_id
    ]
    selected.sort(
        key=lambda event: (int(event.get("step_index", -1)), str(event.get("event_type", "")))
    )

    steps = [event for event in selected if event.get("event_type") == "step"]
    steps.sort(key=lambda event: int(event.get("step_index", -1)))

    terminal = next(
        (event for event in reversed(selected) if event.get("event_type") == "terminal"),
        None,
    )
    return EpisodeReplay(
        episode_id=episode_id,
        events=tuple(selected),
        step_events=tuple(steps),
        terminal_event=terminal,
    )


def summarize_episodes(events: Iterable[dict[str, Any]]) -> tuple[EpisodeSummary, ...]:
    """Compute per-episode summary metrics from event stream."""
    grouped = group_events_by_episode(events)
    summaries: list[EpisodeSummary] = []
    for episode_id, episode_events in grouped.items():
        steps = [event for event in episode_events if event.get("event_type") == "step"]
        total_reward = 0.0
        done_seen = False
        for step in steps:
            reward = step.get("reward", 0.0)
            try:
                total_reward += float(reward)
            except (TypeError, ValueError):
                pass
            if bool(step.get("done", False)):
                done_seen = True

        terminal_event = next(
            (event for event in reversed(episode_events) if event.get("event_type") == "terminal"),
            None,
        )
        terminal_reason = None
        if terminal_event is not None:
            reason = terminal_event.get("reason")
            if isinstance(reason, str):
                terminal_reason = reason

        timestamps = [
            event.get("timestamp_utc")
            for event in episode_events
            if isinstance(event.get("timestamp_utc"), str)
        ]
        first_ts = min(timestamps) if timestamps else None
        last_ts = max(timestamps) if timestamps else None

        summaries.append(
            EpisodeSummary(
                episode_id=episode_id,
                step_count=len(steps),
                total_reward=total_reward,
                done_seen=done_seen,
                terminal_reason=terminal_reason,
                first_timestamp_utc=first_ts,
                last_timestamp_utc=last_ts,
            )
        )

    summaries.sort(key=lambda summary: summary.episode_id)
    return tuple(summaries)


def load_episode_replay(events_path: Path, episode_id: str) -> EpisodeReplay:
    """Load JSONL events from disk and reconstruct one episode."""
    return replay_episode(load_jsonl_events(events_path), episode_id=episode_id)


def load_and_summarize(events_path: Path) -> tuple[EpisodeSummary, ...]:
    """Load JSONL events from disk and summarize all episodes."""
    return summarize_episodes(load_jsonl_events(events_path))
