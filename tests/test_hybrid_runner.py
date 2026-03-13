"""Tests for hybrid runner parser and argument validation."""

from __future__ import annotations

import pytest

from src.hybrid.rewards import HybridMetaRewardWeights
from src.hybrid.runner import (
    _build_meta_reward_weights,
    _build_parser,
    _format_monitor_action_line,
    _validate_args,
)
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
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)
    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)


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
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)
    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)


def test_hybrid_parser_train_full_meta_reward_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-full-hierarchical"])

    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)


def test_hybrid_parser_eval_meta_reward_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["eval-hybrid", "--checkpoint", "artifacts/hybrid/20260311-01-test"])

    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)


def test_hybrid_parser_train_full_requires_warmstart_when_not_resuming() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-full-hierarchical"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_hybrid_validate_args_rejects_missing_restore_save_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(["movement-test", "--restore-save-file", "does-not-exist.bin"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_hybrid_validate_args_rejects_negative_restore_save_delay() -> None:
    parser = _build_parser()
    args = parser.parse_args(["movement-test", "--restore-save-delay", "-0.01"])

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


def test_hybrid_parser_accepts_custom_meta_reward_values() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-meta-no-enemies",
            "--meta-reward-objective-complete",
            "2.25",
            "--meta-reward-phase-progress",
            "0.5",
            "--meta-reward-step-cost",
            "0.02",
            "--meta-reward-premature-exit-penalty",
            "3.0",
            "--meta-reward-sector-advance",
            "1.75",
            "--meta-reward-final-sector-win",
            "40.0",
        ]
    )

    assert args.meta_reward_objective_complete == pytest.approx(2.25)
    assert args.meta_reward_phase_progress == pytest.approx(0.5)
    assert args.meta_reward_step_cost == pytest.approx(0.02)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(3.0)
    assert args.meta_reward_sector_advance == pytest.approx(1.75)
    assert args.meta_reward_final_sector_win == pytest.approx(40.0)


def test_build_meta_reward_weights_uses_cli_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "movement-test",
            "--meta-reward-objective-complete",
            "2.0",
            "--meta-reward-phase-progress",
            "0.4",
            "--meta-reward-step-cost",
            "0.03",
            "--meta-reward-premature-exit-penalty",
            "2.2",
            "--meta-reward-sector-advance",
            "1.3",
            "--meta-reward-final-sector-win",
            "30.0",
        ]
    )

    weights = _build_meta_reward_weights(args)
    assert weights == HybridMetaRewardWeights(
        objective_complete=2.0,
        phase_progress=0.4,
        step_cost=0.03,
        premature_exit_penalty=2.2,
        sector_advance=1.3,
        final_sector_win=30.0,
    )


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
