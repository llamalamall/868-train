"""Tests for hybrid runner parser and argument validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hybrid.rewards import HybridMetaRewardWeights, HybridRewardSuite, HybridThreatRewardWeights
from src.hybrid.runner import (
    HybridEpisodeSummary,
    _build_hybrid_config_payload,
    _build_meta_reward_weights,
    _build_threat_reward_weights,
    _build_parser,
    _build_training_state_payload,
    _classify_terminal_reason,
    _format_monitor_action_line,
    _format_monitor_actions,
    _format_monitor_training_line,
    _hook_event_is_victory_signal,
    _run_rollouts,
    _validate_args,
)
from src.hybrid.types import (
    HybridDecision,
    HybridDecisionTrace,
    MetaObjectiveChoice,
    ObjectivePhase,
    ThreatOverride,
)
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, InventoryState, MapState


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")  # type: ignore[arg-type]


def _runner_snapshot(*, health: int = 10, victory: bool = False) -> GameStateSnapshot:
    extra_fields = {"victory_active": _ok_field(True)} if victory else {}
    return GameStateSnapshot(
        timestamp_utc="2026-03-15T00:00:00Z",
        health=_ok_field(health),
        energy=_ok_field(5),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        inventory=InventoryState(status="ok", raw_prog_ids=()),
        map=MapState(status="missing"),
        extra_fields=extra_fields,
    )


class _NullTui:
    def wait_for_step_gate(self, *, training_line: str, action_line: str) -> None:
        return

    def update(
        self,
        *,
        training_line: str,
        action_line: str,
        reward_line: str | None = None,
        next_available_actions_line: str | None = None,
    ) -> None:
        return

    def consume_manual_step_flag(self) -> bool:
        return False


@pytest.fixture
def _null_tui() -> _NullTui:
    return _NullTui()


class _StubUpdateResult:
    def __init__(self, *, did_update: bool = False) -> None:
        self.did_update = did_update


class _StubMetaObserver:
    epsilon = 0.0

    def __init__(self) -> None:
        self.observe_calls: list[dict[str, object]] = []

    def observe(self, **kwargs: object) -> _StubUpdateResult:
        self.observe_calls.append(dict(kwargs))
        return _StubUpdateResult()


class _StubThreatObserver:
    epsilon = 0.0

    def __init__(self) -> None:
        self.observe_calls = 0

    def observe(self, **_: object) -> _StubUpdateResult:
        self.observe_calls += 1
        return _StubUpdateResult()


class _StubRolloutCoordinator:
    def __init__(self) -> None:
        self.meta_controller = _StubMetaObserver()
        self.threat_controller = _StubThreatObserver()
        self.phase_lock_overrides = 0
        self.stall_releases = 0

    def start_episode(self) -> None:
        return

    def decide(
        self,
        *,
        state: GameStateSnapshot,
        available_actions: tuple[str, ...],
        use_meta_controller: bool,
        use_threat_controller: bool,
        explore_meta: bool,
        explore_threat: bool,
    ) -> HybridDecisionTrace:
        return HybridDecisionTrace(
            decision=HybridDecision(
                objective=MetaObjectiveChoice(
                    phase=ObjectivePhase.COLLECT_SIPHONS,
                    target_position=None,
                    reason="stub",
                ),
                threat_override=ThreatOverride.ROUTE_DEFAULT,
                action=available_actions[0],
                used_fallback=False,
                reason="stub",
            ),
            requested_phase=ObjectivePhase.COLLECT_SIPHONS,
            meta_features=(),
            threat_features=(),
            objective_distance_before=None,
            threat_active=False,
            available_actions=available_actions,
        )

    def observe_step_result(
        self,
        *,
        trace: HybridDecisionTrace,
        info: dict[str, object] | None = None,
        next_state: GameStateSnapshot | None = None,
    ) -> None:
        return

    def allowed_meta_phases(self, state: GameStateSnapshot) -> tuple[ObjectivePhase, ...]:
        return (ObjectivePhase.COLLECT_SIPHONS,)

    def resolve_objective_for_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool = False,
    ) -> tuple[ObjectivePhase, GridPosition | None]:
        return (phase, None)

    def meta_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        scripted_phase: ObjectivePhase,
        target: GridPosition | None,
    ) -> tuple[float, ...]:
        return ()

    def objective_distance(
        self,
        *,
        state: GameStateSnapshot,
        target: GridPosition | None,
    ) -> int | None:
        return None

    def threat_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        route_action: str | None,
        objective_distance: int | None,
    ) -> tuple[float, ...]:
        return ()


class _OverrideRolloutCoordinator(_StubRolloutCoordinator):
    def decide(
        self,
        *,
        state: GameStateSnapshot,
        available_actions: tuple[str, ...],
        use_meta_controller: bool,
        use_threat_controller: bool,
        explore_meta: bool,
        explore_threat: bool,
    ) -> HybridDecisionTrace:
        return HybridDecisionTrace(
            decision=HybridDecision(
                objective=MetaObjectiveChoice(
                    phase=ObjectivePhase.EXIT_SECTOR,
                    target_position=None,
                    reason="override",
                ),
                threat_override=ThreatOverride.ROUTE_DEFAULT,
                action=available_actions[0],
                used_fallback=False,
                reason="override",
            ),
            requested_phase=ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
            meta_features=(1.0, 2.0),
            threat_features=(),
            objective_distance_before=None,
            threat_active=False,
            available_actions=available_actions,
        )

    def allowed_meta_phases(self, state: GameStateSnapshot) -> tuple[ObjectivePhase, ...]:
        return (
            ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
            ObjectivePhase.EXIT_SECTOR,
        )

    def meta_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        scripted_phase: ObjectivePhase,
        target: GridPosition | None,
    ) -> tuple[float, ...]:
        return (3.0, 4.0)


class _SingleStepEnv:
    action_space = ("wait",)

    def __init__(self) -> None:
        self.current_episode_id = "episode-00001"

    def reset(self) -> GameStateSnapshot:
        return _runner_snapshot()

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, object]]:
        return (
            _runner_snapshot(victory=True),
            0.0,
            True,
            {
                "terminal_reason": "state:victory",
                "victory_detected": True,
            },
        )

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        return ("wait",)


def test_hybrid_parser_movement_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["movement-test"])

    assert args.command == "movement-test"
    assert args.episodes == 5
    assert args.max_steps == 250
    assert args.no_enemies is True
    assert args.tui is True
    assert args.window_input is False
    assert args.focus_window is True
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
    assert args.meta_reward_currency_gain == pytest.approx(0.10)
    assert args.meta_reward_energy_gain == pytest.approx(0.10)
    assert args.meta_reward_score_gain == pytest.approx(0.02)
    assert args.meta_reward_prog_gain == pytest.approx(1.50)
    assert args.meta_reward_step_limit_penalty == pytest.approx(5.00)
    assert args.meta_reward_stagnation_penalty == pytest.approx(0.05)
    assert args.meta_reward_stagnation_grace_steps == 3


def test_hybrid_parser_train_meta_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-meta-no-enemies"])

    assert args.command == "train-meta-no-enemies"
    assert args.episodes == 120
    assert args.max_steps == 350
    assert args.no_enemies is True
    assert args.window_input is False
    assert args.focus_window is True
    assert args.meta_epsilon_start == pytest.approx(0.60)
    assert args.game_tick_ms == 1
    assert args.post_action_delay == pytest.approx(0.01)
    assert args.post_action_delay_backoff == pytest.approx(0.02)
    assert args.action_ack_timeout_backoff == pytest.approx(0.10)
    assert args.action_ack_backoff_max_level == 3
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
    assert args.meta_reward_currency_gain == pytest.approx(0.10)
    assert args.meta_reward_energy_gain == pytest.approx(0.10)
    assert args.meta_reward_score_gain == pytest.approx(0.02)
    assert args.meta_reward_prog_gain == pytest.approx(1.50)
    assert args.meta_reward_step_limit_penalty == pytest.approx(5.00)
    assert args.meta_reward_stagnation_penalty == pytest.approx(0.05)
    assert args.meta_reward_stagnation_grace_steps == 3
    assert args.meta_phase_override_credit_mode == "skip_overridden"
    assert args.phase_lock_min_steps == 6
    assert args.target_stall_release_steps == 4
    assert args.victory_monitor is True
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)


def test_hybrid_parser_train_full_meta_reward_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-full-hierarchical"])

    assert args.window_input is False
    assert args.focus_window is True
    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)
    assert args.meta_reward_currency_gain == pytest.approx(0.10)
    assert args.meta_reward_energy_gain == pytest.approx(0.10)
    assert args.meta_reward_score_gain == pytest.approx(0.02)
    assert args.meta_reward_prog_gain == pytest.approx(1.50)
    assert args.meta_reward_step_limit_penalty == pytest.approx(5.00)
    assert args.meta_reward_stagnation_penalty == pytest.approx(0.05)
    assert args.meta_reward_stagnation_grace_steps == 3
    assert args.meta_phase_override_credit_mode == "skip_overridden"
    assert args.victory_monitor is True
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)


def test_hybrid_parser_eval_meta_reward_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["eval-hybrid", "--checkpoint", "artifacts/hybrid/20260311-01-test"])

    assert args.window_input is False
    assert args.focus_window is True
    assert args.meta_reward_objective_complete == pytest.approx(1.50)
    assert args.meta_reward_phase_progress == pytest.approx(0.25)
    assert args.meta_reward_step_cost == pytest.approx(0.01)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(1.25)
    assert args.meta_reward_sector_advance == pytest.approx(1.00)
    assert args.meta_reward_final_sector_win == pytest.approx(25.00)
    assert args.meta_reward_currency_gain == pytest.approx(0.10)
    assert args.meta_reward_energy_gain == pytest.approx(0.10)
    assert args.meta_reward_score_gain == pytest.approx(0.02)
    assert args.meta_reward_prog_gain == pytest.approx(1.50)
    assert args.meta_reward_step_limit_penalty == pytest.approx(5.00)
    assert args.meta_reward_stagnation_penalty == pytest.approx(0.05)
    assert args.meta_reward_stagnation_grace_steps == 3


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


def test_hybrid_parser_train_full_accepts_warmstart_path(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "warmstart"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "meta_controller.pt").write_text("meta", encoding="utf-8")
    (checkpoint_dir / "hybrid_config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "training_state.json").write_text("{}", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-full-hierarchical",
            "--warmstart-checkpoint",
            str(checkpoint_dir),
            "--episodes",
            "30",
        ]
    )

    _validate_args(parser, args)
    assert args.warmstart_checkpoint == str(checkpoint_dir)
    assert args.episodes == 30


def test_hybrid_validate_args_rejects_incomplete_warmstart_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_dir = tmp_path / "incomplete-run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "victory-transition-events.jsonl").write_text("", encoding="utf-8")
    (checkpoint_dir / "victory-transition-monitor.log").write_text("", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-full-hierarchical",
            "--warmstart-checkpoint",
            str(checkpoint_dir),
        ]
    )

    with pytest.raises(SystemExit):
        _validate_args(parser, args)

    captured = capsys.readouterr()
    assert "--warmstart-checkpoint is incomplete" in captured.err
    assert "Missing files: meta_controller.pt, hybrid_config.json, training_state.json." in captured.err


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
            "--meta-reward-currency-gain",
            "0.3",
            "--meta-reward-energy-gain",
            "0.2",
            "--meta-reward-score-gain",
            "0.05",
            "--meta-reward-prog-gain",
            "3.5",
            "--meta-reward-step-limit-penalty",
            "7.0",
            "--meta-reward-stagnation-penalty",
            "0.15",
            "--meta-reward-stagnation-grace-steps",
            "5",
        ]
    )

    assert args.meta_reward_objective_complete == pytest.approx(2.25)
    assert args.meta_reward_phase_progress == pytest.approx(0.5)
    assert args.meta_reward_step_cost == pytest.approx(0.02)
    assert args.meta_reward_premature_exit_penalty == pytest.approx(3.0)
    assert args.meta_reward_sector_advance == pytest.approx(1.75)
    assert args.meta_reward_final_sector_win == pytest.approx(40.0)
    assert args.meta_reward_currency_gain == pytest.approx(0.3)
    assert args.meta_reward_energy_gain == pytest.approx(0.2)
    assert args.meta_reward_score_gain == pytest.approx(0.05)
    assert args.meta_reward_prog_gain == pytest.approx(3.5)
    assert args.meta_reward_step_limit_penalty == pytest.approx(7.0)
    assert args.meta_reward_stagnation_penalty == pytest.approx(0.15)
    assert args.meta_reward_stagnation_grace_steps == 5


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
            "--meta-reward-currency-gain",
            "0.4",
            "--meta-reward-energy-gain",
            "0.25",
            "--meta-reward-score-gain",
            "0.08",
            "--meta-reward-prog-gain",
            "2.75",
            "--meta-reward-step-limit-penalty",
            "4.5",
            "--meta-reward-stagnation-penalty",
            "0.12",
            "--meta-reward-stagnation-grace-steps",
            "6",
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
        currency_gain=0.4,
        energy_gain=0.25,
        score_gain=0.08,
        prog_gain=2.75,
        step_limit_penalty=4.5,
        stagnation_penalty=0.12,
        stagnation_grace_steps=6,
    )


def test_build_threat_reward_weights_uses_cli_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-full-hierarchical",
            "--warmstart-checkpoint",
            "artifacts/hybrid/meta",
            "--threat-reward-survival",
            "0.08",
            "--threat-reward-damage-taken-penalty",
            "0.60",
            "--threat-reward-fail-penalty",
            "4.5",
            "--threat-reward-route-rejoin-bonus",
            "0.0",
            "--threat-reward-invalid-override-penalty",
            "0.25",
            "--threat-reward-enemy-damaged",
            "0.4",
            "--threat-reward-enemy-cleared",
            "1.5",
            "--threat-reward-spawn-debt-penalty",
            "0.3",
        ]
    )

    weights = _build_threat_reward_weights(args)
    assert weights == HybridThreatRewardWeights(
        survival=0.08,
        damage_taken_penalty=0.60,
        fail_penalty=4.5,
        route_rejoin_bonus=0.0,
        invalid_override_penalty=0.25,
        enemy_damaged=0.4,
        enemy_cleared=1.5,
        spawn_debt_penalty=0.3,
    )


def test_build_hybrid_config_payload_includes_run_tag_and_warmstart_metadata() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-full-hierarchical",
            "--warmstart-checkpoint",
            "artifacts/hybrid/20260314-03-hybrid-beta",
            "--run-tag",
            "hybrid-full-beta-fixedmeta",
            "--episodes",
            "320",
            "--no-joint-finetune",
        ]
    )

    payload = _build_hybrid_config_payload(
        args,
        command="train-full-hierarchical",
        restore_save_source=None,
        meta_reward_weights=_build_meta_reward_weights(args),
        threat_reward_weights=_build_threat_reward_weights(args),
        victory_monitor_enabled=True,
        victory_monitor_output_path=Path("artifacts/hybrid/20260315-09-test/victory-transition-events.jsonl"),
        victory_monitor_log_path=Path("artifacts/hybrid/20260315-09-test/victory-transition-monitor.log"),
    )

    assert payload["run_tag"] == "hybrid-full-beta-fixedmeta"
    assert payload["victory_monitor_enabled"] is True
    assert payload["victory_monitor_output_path"].endswith("victory-transition-events.jsonl")
    assert payload["victory_monitor_log_path"].endswith("victory-transition-monitor.log")
    assert payload["warmstart_checkpoint"] == "artifacts/hybrid/20260314-03-hybrid-beta"
    assert payload["resume_checkpoint"] is None
    assert payload["post_action_delay"] == pytest.approx(0.01)
    assert payload["action_ack_timeout"] == pytest.approx(0.35)
    assert payload["action_ack_poll_interval"] == pytest.approx(0.05)
    assert payload["post_action_delay_backoff"] == pytest.approx(0.02)
    assert payload["action_ack_timeout_backoff"] == pytest.approx(0.10)
    assert payload["action_ack_backoff_max_level"] == 3
    assert payload["phase_lock_min_steps"] == 6
    assert payload["target_stall_release_steps"] == 4
    assert payload["meta_phase_override_credit_mode"] == "skip_overridden"
    assert payload["meta_reward_weights"]["final_sector_win"] == pytest.approx(25.0)
    assert payload["meta_reward_weights"]["currency_gain"] == pytest.approx(0.10)
    assert payload["meta_reward_weights"]["energy_gain"] == pytest.approx(0.10)
    assert payload["meta_reward_weights"]["score_gain"] == pytest.approx(0.02)
    assert payload["meta_reward_weights"]["prog_gain"] == pytest.approx(1.50)
    assert payload["meta_reward_weights"]["step_limit_penalty"] == pytest.approx(5.0)
    assert payload["meta_reward_weights"]["stagnation_penalty"] == pytest.approx(0.05)
    assert payload["meta_reward_weights"]["stagnation_grace_steps"] == 3
    assert payload["threat_reward_weights"]["survival"] == pytest.approx(0.05)
    assert payload["threat_reward_weights"]["damage_taken_penalty"] == pytest.approx(0.35)
    assert payload["threat_reward_weights"]["fail_penalty"] == pytest.approx(2.5)
    assert payload["threat_reward_weights"]["enemy_damaged"] == pytest.approx(0.20)
    assert payload["threat_reward_weights"]["enemy_cleared"] == pytest.approx(0.75)
    assert payload["threat_reward_weights"]["spawn_debt_penalty"] == pytest.approx(0.15)


def test_build_hybrid_config_payload_records_resume_checkpoint_for_meta_training() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-meta-no-enemies",
            "--resume-checkpoint",
            "artifacts/hybrid/20260314-03-hybrid-beta",
            "--run-tag",
            "hybrid-meta-beta-efficient",
        ]
    )

    payload = _build_hybrid_config_payload(
        args,
        command="train-meta-no-enemies",
        restore_save_source=None,
        meta_reward_weights=_build_meta_reward_weights(args),
        threat_reward_weights=_build_threat_reward_weights(args),
        victory_monitor_enabled=False,
        victory_monitor_output_path=None,
        victory_monitor_log_path=None,
    )

    assert payload["run_tag"] == "hybrid-meta-beta-efficient"
    assert payload["victory_monitor_enabled"] is False
    assert payload["victory_monitor_output_path"] is None
    assert payload["victory_monitor_log_path"] is None
    assert payload["warmstart_checkpoint"] is None
    assert payload["resume_checkpoint"] == "artifacts/hybrid/20260314-03-hybrid-beta"


def test_build_training_state_payload_includes_summary_and_extended_episode_fields() -> None:
    results = (
        HybridEpisodeSummary(
            episode_id="episode-00001",
            steps=320,
            done=False,
            terminal_reason=None,
            total_reward=18.0,
            meta_reward_total=12.0,
            threat_reward_total=6.0,
            invalid_actions=5,
            premature_exit_attempts=0,
            route_length_total=180,
            route_replans=7,
            hit_step_limit=True,
            terminal_classification="step_limit",
            phase_switches=4,
            threat_active_steps=9,
            threat_rewarded_steps=0,
            route_rejoin_events=0,
            phase_lock_overrides=2,
            stall_releases=1,
            requested_phase_override_steps=2,
            requested_phase_counts={"collect_siphons": 1, "exit_sector": 1},
            executed_phase_counts={"collect_siphons": 2},
            requested_executed_phase_pair_counts={
                "collect_siphons->collect_siphons": 1,
                "exit_sector->collect_siphons": 1,
            },
            meta_override_updates_skipped=1,
            invalid_action_reason_counts={"action_not_acknowledged": 2},
            action_ack_reason_counts={"no_observed_effect": 2, "state_changed": 3},
            invalid_action_action_counts={"move_right": 2},
            action_ack_timeout_action_counts={"move_right": 2},
            start_screen_detected=False,
            victory_detected=False,
            victory_inferred_from_start_screen=False,
            death_to_start_screen_detected=False,
            start_screen_unknown_detected=False,
            final_action=None,
            last_known_player_position=None,
            last_known_exit_position=None,
        ),
        HybridEpisodeSummary(
            episode_id="episode-00002",
            steps=210,
            done=True,
            terminal_reason="state:start_screen_unknown",
            total_reward=24.0,
            meta_reward_total=18.0,
            threat_reward_total=6.0,
            invalid_actions=2,
            premature_exit_attempts=1,
            route_length_total=120,
            route_replans=3,
            hit_step_limit=False,
            terminal_classification="unexpected_start_screen",
            phase_switches=2,
            threat_active_steps=0,
            threat_rewarded_steps=1,
            route_rejoin_events=1,
            phase_lock_overrides=0,
            stall_releases=0,
            requested_phase_override_steps=0,
            requested_phase_counts={"exit_sector": 2},
            executed_phase_counts={"exit_sector": 2},
            requested_executed_phase_pair_counts={"exit_sector->exit_sector": 2},
            meta_override_updates_skipped=0,
            invalid_action_reason_counts={"prog_no_effect": 1},
            action_ack_reason_counts={"state_changed": 2},
            invalid_action_action_counts={"prog_slot_1": 1},
            action_ack_timeout_action_counts={},
            start_screen_detected=True,
            victory_detected=False,
            victory_inferred_from_start_screen=False,
            death_to_start_screen_detected=False,
            start_screen_unknown_detected=True,
            final_action="move_right",
            last_known_player_position={"x": 4, "y": 5},
            last_known_exit_position={"x": 5, "y": 5},
        ),
    )

    payload = _build_training_state_payload(
        results=results,
        episodes_requested=2,
        max_steps=320,
        saved_at_utc="2026-03-14T15:00:00Z",
    )

    assert payload["episodes_requested"] == 2
    assert payload["episodes_completed"] == 2
    assert payload["max_steps_per_episode"] == 320
    assert payload["saved_at_utc"] == "2026-03-14T15:00:00Z"
    assert payload["summary"]["episodes"] == 2
    assert payload["summary"]["done_episodes"] == 1
    assert payload["summary"]["non_death_terminal_episodes"] == 0
    assert payload["summary"]["unexpected_start_screen_episodes"] == 1
    assert payload["summary"]["hit_step_limit_episodes"] == 1
    assert payload["summary"]["non_death_terminal_rate"] == pytest.approx(0.0)
    assert payload["summary"]["unexpected_start_screen_rate"] == pytest.approx(0.5)
    assert payload["summary"]["hit_step_limit_rate"] == pytest.approx(0.5)
    assert payload["summary"]["avg_threat_rewarded_steps"] == pytest.approx(0.5)
    assert payload["summary"]["avg_route_rejoin_events"] == pytest.approx(0.5)
    assert payload["summary"]["avg_phase_lock_overrides"] == pytest.approx(1.0)
    assert payload["summary"]["avg_stall_releases"] == pytest.approx(0.5)
    assert payload["summary"]["avg_requested_phase_override_steps"] == pytest.approx(1.0)
    assert payload["summary"]["avg_meta_override_updates_skipped"] == pytest.approx(0.5)
    assert payload["summary"]["terminal_reason_counts"] == {
        "none": 1,
        "state:start_screen_unknown": 1,
    }
    assert payload["summary"]["terminal_classification_counts"] == {
        "step_limit": 1,
        "unexpected_start_screen": 1,
    }
    assert payload["summary"]["requested_phase_counts"] == {
        "collect_siphons": 1,
        "exit_sector": 3,
    }
    assert payload["summary"]["executed_phase_counts"] == {
        "collect_siphons": 2,
        "exit_sector": 2,
    }
    assert payload["summary"]["requested_executed_phase_pair_counts"] == {
        "collect_siphons->collect_siphons": 1,
        "exit_sector->collect_siphons": 1,
        "exit_sector->exit_sector": 2,
    }
    assert payload["summary"]["invalid_action_reason_counts"] == {
        "action_not_acknowledged": 2,
        "prog_no_effect": 1,
    }
    assert payload["summary"]["action_ack_reason_counts"] == {
        "no_observed_effect": 2,
        "state_changed": 5,
    }
    assert payload["summary"]["invalid_action_action_counts"] == {
        "move_right": 2,
        "prog_slot_1": 1,
    }
    assert payload["summary"]["action_ack_timeout_action_counts"] == {
        "move_right": 2,
    }
    assert payload["results"][0]["hit_step_limit"] is True
    assert payload["results"][0]["terminal_classification"] == "step_limit"
    assert payload["results"][0]["phase_switches"] == 4
    assert payload["results"][0]["threat_active_steps"] == 9
    assert payload["results"][0]["phase_lock_overrides"] == 2
    assert payload["results"][0]["requested_phase_override_steps"] == 2
    assert payload["results"][0]["meta_override_updates_skipped"] == 1
    assert payload["results"][0]["requested_executed_phase_pair_counts"] == {
        "collect_siphons->collect_siphons": 1,
        "exit_sector->collect_siphons": 1,
    }
    assert payload["results"][0]["invalid_action_reason_counts"] == {
        "action_not_acknowledged": 2,
    }
    assert payload["results"][0]["invalid_action_action_counts"] == {
        "move_right": 2,
    }
    assert payload["results"][0]["action_ack_timeout_action_counts"] == {
        "move_right": 2,
    }
    assert payload["results"][1]["terminal_classification"] == "unexpected_start_screen"
    assert payload["results"][1]["start_screen_unknown_detected"] is True
    assert payload["results"][1]["final_action"] == "move_right"
    assert payload["results"][1]["last_known_player_position"] == {"x": 4, "y": 5}
    assert payload["results"][1]["last_known_exit_position"] == {"x": 5, "y": 5}
    assert payload["results"][1]["requested_phase_counts"] == {"exit_sector": 2}
    assert payload["results"][1]["meta_override_updates_skipped"] == 0
    assert payload["results"][1]["action_ack_reason_counts"] == {"state_changed": 2}
    assert payload["results"][1]["invalid_action_action_counts"] == {"prog_slot_1": 1}


def test_hook_event_is_victory_signal_matches_victory_flag_targets() -> None:
    normal_victory_event = {"target_name": "normal_victory_flag_set", "snapshot": {}}
    points_victory_event = {"target_name": "points_victory_flag_set", "snapshot": {}}
    unrelated_event = {"target_name": "game_over_flag_set", "snapshot": {}}

    assert _hook_event_is_victory_signal(normal_victory_event) is True
    assert _hook_event_is_victory_signal(points_victory_event) is True
    assert _hook_event_is_victory_signal(unrelated_event) is False


def test_classify_terminal_reason_treats_start_screen_unknown_as_unexpected() -> None:
    assert _classify_terminal_reason(
        done=True,
        terminal_reason="state:start_screen_unknown",
        hit_step_limit=False,
    ) == "unexpected_start_screen"
    assert _classify_terminal_reason(
        done=True,
        terminal_reason="state:sector_exit_to_start_screen",
        hit_step_limit=False,
    ) == "non_death_terminal"
    assert _classify_terminal_reason(
        done=True,
        terminal_reason="state:death_to_start_screen",
        hit_step_limit=False,
    ) == "fail_or_death"


def test_run_rollouts_does_not_add_threat_reward_when_threat_controller_is_disabled(
    _null_tui: _NullTui,
) -> None:
    coordinator = _StubRolloutCoordinator()
    results = _run_rollouts(
        env=_SingleStepEnv(),
        coordinator=coordinator,
        reward_suite=HybridRewardSuite(),
        episodes=1,
        max_steps=5,
        train_meta=False,
        train_threat=False,
        use_meta=False,
        use_threat=False,
        explore_meta=False,
        explore_threat=False,
        tui=_null_tui,
        monitor_enabled=False,
        print_reward_breakdown=False,
    )

    assert results[0].terminal_reason == "state:victory"
    assert results[0].threat_reward_total == pytest.approx(0.0)
    assert results[0].threat_rewarded_steps == 0
    assert results[0].route_rejoin_events == 0
    assert results[0].final_action == "wait"
    assert results[0].last_known_player_position is None
    assert results[0].last_known_exit_position is None
    assert coordinator.threat_controller.observe_calls == 0


def test_run_rollouts_skips_meta_updates_for_overridden_requested_phase(
    _null_tui: _NullTui,
) -> None:
    coordinator = _OverrideRolloutCoordinator()

    results = _run_rollouts(
        env=_SingleStepEnv(),
        coordinator=coordinator,
        reward_suite=HybridRewardSuite(),
        episodes=1,
        max_steps=5,
        train_meta=True,
        train_threat=False,
        use_meta=True,
        use_threat=False,
        explore_meta=False,
        explore_threat=False,
        tui=_null_tui,
        monitor_enabled=False,
        print_reward_breakdown=False,
        meta_phase_override_credit_mode="skip_overridden",
    )

    assert coordinator.meta_controller.observe_calls == []
    assert results[0].requested_phase_override_steps == 1
    assert results[0].meta_override_updates_skipped == 1


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


def test_hybrid_format_monitor_actions_preserves_full_action_list() -> None:
    formatted = _format_monitor_actions(
        ("move_up", "move_down", "move_left", "move_right", "space", "prog_slot_10"),
        limit=2,
    )

    assert formatted == "move_up,move_down,move_left,move_right,space,prog_slot_10"


def test_format_monitor_training_line_includes_threat_epsilon_when_available() -> None:
    line = _format_monitor_training_line(
        episode_id="episode-00001",
        step=7,
        reward=1.25,
        total_reward=3.5,
        meta_epsilon=0.4,
        threat_epsilon=0.2,
        updates_applied=9,
        done=False,
        terminal_reason=None,
    )

    assert line == (
        "episode=episode-00001 step=7 reward=1.250 total=3.500 epsilon=0.4000 "
        "threat_epsilon=0.2000 updates=9 done=False terminal=-"
    )
