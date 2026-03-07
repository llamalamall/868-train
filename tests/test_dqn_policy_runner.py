"""Tests for DQN policy runner CLI wiring helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.env.dqn_policy_runner import (
    _build_dqn_config,
    _build_parser,
    _default_checkpoint_path,
    _resolve_checkpoint_path,
    _validate_args,
)


def test_dqn_runner_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.mode == "train"
    assert args.exe == "868-HACK.exe"
    assert args.episodes == 10
    assert args.max_steps == 200
    assert args.checkpoint is None
    assert args.checkpoint_every == 0
    assert args.movement_keys == "arrows"
    assert args.prog_actions is True
    assert args.gamma == 0.99
    assert args.learning_rate == 0.01


def test_dqn_runner_parser_accepts_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "eval",
            "--episodes",
            "3",
            "--max-steps",
            "90",
            "--checkpoint",
            "artifacts/checkpoints/final.json",
            "--movement-keys",
            "numpad",
            "--no-prog-actions",
            "--gamma",
            "0.9",
            "--learning-rate",
            "0.02",
            "--target-sync-interval",
            "50",
        ]
    )

    assert args.mode == "eval"
    assert args.episodes == 3
    assert args.max_steps == 90
    assert args.checkpoint == "artifacts/checkpoints/final.json"
    assert args.movement_keys == "numpad"
    assert args.prog_actions is False
    assert args.gamma == 0.9
    assert args.learning_rate == 0.02
    assert args.target_sync_interval == 50


def test_default_checkpoint_path_uses_artifacts_checkpoints_dir() -> None:
    path = _default_checkpoint_path(
        now_utc=datetime(2026, 3, 6, 12, 34, 56, tzinfo=timezone.utc)
    )

    assert path == Path("artifacts") / "checkpoints" / "dqn-20260306-123456.json"


def test_resolve_checkpoint_path_prefers_explicit_value() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--checkpoint", "artifacts/checkpoints/custom.json"])

    assert _resolve_checkpoint_path(args) == Path("artifacts/checkpoints/custom.json")


def test_build_dqn_config_maps_cli_values() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--gamma",
            "0.93",
            "--learning-rate",
            "0.015",
            "--replay-capacity",
            "777",
            "--min-replay-size",
            "22",
            "--batch-size",
            "11",
            "--target-sync-interval",
            "17",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.2",
            "--epsilon-decay-steps",
            "333",
        ]
    )

    config = _build_dqn_config(args)

    assert config.gamma == 0.93
    assert config.learning_rate == 0.015
    assert config.replay_capacity == 777
    assert config.min_replay_size == 22
    assert config.batch_size == 11
    assert config.target_sync_interval == 17
    assert config.epsilon_start == 0.8
    assert config.epsilon_end == 0.2
    assert config.epsilon_decay_steps == 333


def test_validate_args_requires_checkpoint_in_eval_mode() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--mode", "eval"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_validate_args_rejects_negative_checkpoint_every() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--checkpoint-every", "-1"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)
