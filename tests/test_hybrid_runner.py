"""Tests for hybrid runner parser and argument validation."""

from __future__ import annotations

import pytest

from src.hybrid.runner import _build_parser, _format_monitor_action_line, _validate_args
from src.hybrid.types import ObjectivePhase
from src.state.schema import GridPosition


def test_hybrid_parser_movement_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["movement-test"])

    assert args.command == "movement-test"
    assert args.episodes == 5
    assert args.max_steps == 250
    assert args.no_enemies is True
    assert args.tui is True
    assert args.step_through is False
    assert args.game_tick_ms == 1
    assert args.post_action_delay == pytest.approx(0.01)
    assert args.disable_idle_frame_delay is True
    assert args.disable_background_motion is True
    assert args.disable_wall_animations is True


def test_hybrid_parser_train_meta_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-meta-no-enemies"])

    assert args.command == "train-meta-no-enemies"
    assert args.episodes == 120
    assert args.max_steps == 350
    assert args.no_enemies is True
    assert args.meta_epsilon_start == pytest.approx(0.60)
    assert args.game_tick_ms == 1
    assert args.post_action_delay == pytest.approx(0.01)
    assert args.disable_idle_frame_delay is True
    assert args.disable_background_motion is True
    assert args.disable_wall_animations is True


def test_hybrid_parser_train_full_requires_warmstart_when_not_resuming() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-full-hierarchical"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_hybrid_parser_train_full_accepts_warmstart_path() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-full-hierarchical",
            "--warmstart-checkpoint",
            "artifacts/hybrid/20260311-01-gateb",
            "--episodes",
            "30",
        ]
    )

    _validate_args(parser, args)
    assert args.warmstart_checkpoint == "artifacts/hybrid/20260311-01-gateb"
    assert args.episodes == 30


def test_hybrid_parser_eval_requires_checkpoint() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval-hybrid"])


def test_format_monitor_action_line_includes_phase_and_target_coordinates() -> None:
    line = _format_monitor_action_line(
        action="move_right",
        reason="scripted_phase_only",
        phase=ObjectivePhase.COLLECT_SIPHONS,
        target=GridPosition(x=3, y=4),
    )
    assert line == (
        "action=move_right phase=collect_siphons next_target=(3,4) "
        "reason=scripted_phase_only"
    )
