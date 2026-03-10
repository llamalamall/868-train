"""Tests for gym-like game environment wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.controller.action_api import ActionConfig, ActionTimings
from src.env.game_env import (
    GameEnv,
    GameEnvConfig,
    GameEnvError,
    ResetTimeoutError,
    StateDesyncError,
    StepTimeoutError,
    _clamp_action_press_duration_to_game_tick,
    run_random_policy,
)
from src.env.reset_manager import NoopResetManager
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapState,
)


def _field(value: object, *, status: str = "ok") -> FieldState:
    return FieldState(value=value, status=status)  # type: ignore[arg-type]


def _snapshot(
    *,
    health: int = 10,
    energy: int = 5,
    credits: int = 3,
    failed: bool = False,
    map_state: MapState | None = None,
    inventory_state: InventoryState | None = None,
    can_siphon_now: bool | None = None,
    prog_slots_available_mask: int | None = None,
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=_field(health),
        energy=_field(energy),
        currency=_field(credits),
        fail_state=_field(failed),
        inventory=inventory_state or InventoryState(status="missing"),
        map=map_state or MapState(status="missing"),
        can_siphon_now=can_siphon_now,
        prog_slots_available_mask=prog_slots_available_mask,
    )


def _start_screen_snapshot() -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=FieldState(
            value=None,
            status="invalid",
            error_code="null_pointer",
            error="Pointer chain resolved to null while on start screen.",
            source_field="player_health",
        ),
        energy=_field(0),
        currency=_field(0),
        fail_state=_field(False, status="missing"),
        map=MapState(status="missing"),
    )


def _desynced_snapshot() -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=FieldState(
            value=None,
            status="invalid",
            error_code="read_failed",
            error="ReadProcessMemory failed.",
            source_field="player_health",
        ),
        energy=FieldState(
            value=None,
            status="invalid",
            error_code="short_read",
            error="Short read while extracting energy.",
            source_field="player_energy",
        ),
        currency=_field(0, status="missing"),
        fail_state=_field(False, status="missing"),
        inventory=InventoryState(status="missing"),
        map=MapState(status="missing"),
    )


@dataclass
class FakeActionAPI:
    """Minimal fake action API for env tests."""

    delay_seconds: float = 0.0
    actions: list[str] = field(default_factory=list)

    def perform_action(self, action_name: str) -> None:
        self.actions.append(action_name)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)


@dataclass
class FailingActionAPI(FakeActionAPI):
    failures_remaining: int = 0

    def perform_action(self, action_name: str) -> None:
        self.actions.append(action_name)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("transient input failure")
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)


@dataclass
class FlakyResetStrategy:
    failures_remaining: int = 0
    calls: int = 0

    def reset(self) -> None:
        self.calls += 1
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("transient reset failure")


class QueueStateProvider:
    """Return queued states; once exhausted, keep returning last state."""

    def __init__(self, states: list[GameStateSnapshot]) -> None:
        self._states = list(states)
        self.calls = 0

    def __call__(self) -> GameStateSnapshot:
        self.calls += 1
        if len(self._states) > 1:
            return self._states.pop(0)
        return self._states[0]


@dataclass
class FakeClock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration_seconds: float) -> None:
        self.now += max(float(duration_seconds), 0.0)


def test_clamp_action_press_duration_to_game_tick_caps_long_press() -> None:
    base = ActionConfig(
        timings=ActionTimings(press_duration_seconds=0.05),
    )

    clamped = _clamp_action_press_duration_to_game_tick(
        action_config=base,
        game_tick_ms=8,
    )

    assert clamped.timings.press_duration_seconds == pytest.approx(0.008)


def test_clamp_action_press_duration_to_game_tick_keeps_short_press() -> None:
    base = ActionConfig(
        timings=ActionTimings(press_duration_seconds=0.004),
    )

    clamped = _clamp_action_press_duration_to_game_tick(
        action_config=base,
        game_tick_ms=8,
    )

    assert clamped is base
    assert clamped.timings.press_duration_seconds == pytest.approx(0.004)


def test_game_env_reset_and_step_contract() -> None:
    initial = _snapshot(health=10, failed=False)
    after_step = _snapshot(health=9, failed=False)
    state_provider = QueueStateProvider([initial, after_step])
    actions = FakeActionAPI()

    def reward_fn(previous: GameStateSnapshot, current: GameStateSnapshot, done: bool, _info: dict) -> float:
        assert previous.health.value == 10
        assert current.health.value == 9
        assert done is False
        return 1.5

    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        reward_fn=reward_fn,
        action_space=("move_up", "wait"),
    )

    reset_state = env.reset()
    assert reset_state.health.value == 10
    assert env.current_episode_id is not None

    state, reward, done, info = env.step("move_up")
    assert state.health.value == 9
    assert reward == 1.5
    assert done is False
    assert info["step_index"] == 0
    assert info["action"] == "move_up"
    assert actions.actions == ["move_up"]


def test_game_env_step_performs_implicit_reset_when_episode_not_started() -> None:
    initial = _snapshot(health=10, failed=False)
    after_step = _snapshot(health=8, failed=False)
    state_provider = QueueStateProvider([initial, after_step])
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    _state, _reward, _done, info = env.step("wait")

    assert env.current_episode_id is not None
    assert info["reset_performed"] is True
    assert info["step_index"] == 0


def test_game_env_step_sets_done_from_fail_detector_result() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _snapshot(failed=False)])
    actions = FakeActionAPI()
    fail_detector = SimpleNamespace(
        check=lambda: SimpleNamespace(
            is_terminal=True,
            reason="memory:fail_state",
            source="memory",
            error=None,
        )
    )
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        fail_detector=fail_detector,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
    )

    env.reset()
    _, _, done, info = env.step("wait")

    assert done is True
    assert info["terminal_reason"] == "memory:fail_state"


def test_game_env_step_marks_done_when_fail_detector_reports_null_pointer_start_screen() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _start_screen_snapshot()])
    actions = FakeActionAPI()
    fail_detector = SimpleNamespace(
        check=lambda: SimpleNamespace(
            is_terminal=False,
            reason="memory_unavailable",
            source="memory",
            error="null_pointer",
        )
    )
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        fail_detector=fail_detector,
        reset_strategy=NoopResetManager(),
        action_space=("wait", "confirm", "space"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    env.reset()
    _, _, done, info = env.step("wait")

    assert done is True
    assert info["terminal_reason"] == "state:start_screen"
    assert info["start_screen_detected"] is True


def test_game_env_reset_dispatches_confirm_and_space_when_start_screen_null_pointer_seen() -> None:
    state_provider = QueueStateProvider([_start_screen_snapshot(), _snapshot(failed=False)])
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait", "confirm", "space"),
        config=GameEnvConfig(
            require_non_terminal_on_reset=False,
            state_poll_interval_seconds=0.0,
            post_action_poll_delay_seconds=0.0,
        ),
    )

    state = env.reset()

    assert state.health.status == "ok"
    assert actions.actions == ["confirm", "space"]


def test_game_env_reset_dispatches_recovery_actions_when_terminal_state_persists() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=True), _snapshot(failed=False)])
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait", "confirm", "space"),
        config=GameEnvConfig(
            require_non_terminal_on_reset=True,
            state_poll_interval_seconds=0.0,
            post_action_poll_delay_seconds=0.0,
        ),
    )

    state = env.reset()

    assert state.fail_state.status == "ok"
    assert state.fail_state.value is False
    assert actions.actions == ["confirm", "space"]


def test_game_env_reset_timeout_when_terminal_state_persists() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=True)])
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(
            reset_timeout_seconds=0.05,
            state_timeout_seconds=0.05,
            state_poll_interval_seconds=0.01,
            require_non_terminal_on_reset=True,
        ),
    )

    with pytest.raises(ResetTimeoutError):
        env.reset()


def test_game_env_step_timeout_when_action_blocks() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(delay_seconds=0.20),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(
            step_timeout_seconds=0.05,
            state_timeout_seconds=0.05,
            reset_timeout_seconds=0.05,
            require_non_terminal_on_reset=False,
        ),
    )
    env.reset()
    with pytest.raises(StepTimeoutError):
        env.step("wait")


def test_game_env_step_retries_action_after_transient_input_failure() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _snapshot(failed=False)])
    actions = FailingActionAPI(failures_remaining=1)
    recovery_reasons: list[str] = []
    env = GameEnv(
        action_api=actions,
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False, action_retry_attempts=2),
        recovery_hook=lambda reason, _error: recovery_reasons.append(reason),
    )

    env.reset()
    _state, _reward, done, _info = env.step("wait")

    assert done is False
    assert actions.actions == ["wait", "wait"]
    assert recovery_reasons == ["action_dispatch_failed:wait"]


def test_game_env_waits_for_action_processing_before_returning_step_state() -> None:
    before = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
        ),
    )
    after = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(1, 0),
        ),
    )
    clock = FakeClock()
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([before, before, after]),
        reset_strategy=NoopResetManager(),
        action_space=("move_right",),
        config=GameEnvConfig(
            require_non_terminal_on_reset=False,
            post_action_poll_delay_seconds=0.0,
            wait_for_action_processing=True,
            action_ack_timeout_seconds=0.2,
            action_ack_poll_interval_seconds=0.05,
        ),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    env.reset()

    state, _reward, _done, info = env.step("move_right")

    assert state.map.player_position == GridPosition(1, 0)
    assert info["action_acknowledged"] is True
    assert info["action_ack_reason"] == "state_changed"
    assert int(info["action_ack_checks"]) >= 2


def test_game_env_marks_step_unacknowledged_after_action_ack_timeout() -> None:
    stale = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
        ),
    )
    clock = FakeClock()
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([stale, stale]),
        reset_strategy=NoopResetManager(),
        action_space=("move_right",),
        config=GameEnvConfig(
            require_non_terminal_on_reset=False,
            post_action_poll_delay_seconds=0.0,
            wait_for_action_processing=True,
            action_ack_timeout_seconds=0.12,
            action_ack_poll_interval_seconds=0.05,
        ),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    env.reset()

    _state, _reward, done, info = env.step("move_right")

    assert done is False
    assert info["action_acknowledged"] is False
    assert info["action_ack_reason"] == "action_ack_timeout"
    assert info["action_effective"] is False
    assert info["invalid_action_reason"] == "action_not_acknowledged"


def test_game_env_reset_retries_after_transient_reset_failure() -> None:
    strategy = FlakyResetStrategy(failures_remaining=1)
    recovery_reasons: list[str] = []
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=strategy,
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False, reset_retry_attempts=2),
        recovery_hook=lambda reason, _error: recovery_reasons.append(reason),
    )

    state = env.reset()

    assert state.fail_state.value is False
    assert strategy.calls == 2
    assert recovery_reasons == ["reset_strategy_failed"]


def test_game_env_step_recovers_from_desynced_state_snapshot() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _desynced_snapshot(), _snapshot(failed=False)])
    recovery_reasons: list[str] = []
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False, state_read_retry_attempts=2),
        recovery_hook=lambda reason, _error: recovery_reasons.append(reason),
    )

    env.reset()
    _state, _reward, done, _info = env.step("wait")

    assert done is False
    assert recovery_reasons
    assert recovery_reasons[0].startswith("state_desync:")


def test_game_env_step_raises_state_desync_error_without_recovery_hook() -> None:
    state_provider = QueueStateProvider([_snapshot(failed=False), _desynced_snapshot()])
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=state_provider,
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False, state_read_retry_attempts=1),
    )

    env.reset()
    with pytest.raises(StateDesyncError):
        env.step("wait")


def test_game_env_reports_actionable_runtime_recovery_error() -> None:
    def failing_recovery(_reason: str, _error: Exception | None) -> None:
        raise RuntimeError("reattach failed")

    env = GameEnv(
        action_api=FailingActionAPI(failures_remaining=2),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("wait",),
        config=GameEnvConfig(require_non_terminal_on_reset=False, action_retry_attempts=2),
        recovery_hook=failing_recovery,
    )

    env.reset()
    with pytest.raises(GameEnvError) as error:
        env.step("wait")

    assert "runtime_recovery_failed" in str(error.value)


def test_run_random_policy_returns_episode_results() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    results = run_random_policy(env=env, episodes=2, max_steps_per_episode=2, seed=7)

    assert len(results) == 2
    assert all(result.steps == 2 for result in results)
    assert all(result.done is False for result in results)


def test_run_random_policy_step_callback_reports_action_and_reason() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    events: list[dict[str, object]] = []

    _ = run_random_policy(
        env=env,
        episodes=1,
        max_steps_per_episode=1,
        seed=7,
        step_callback=lambda event: events.append(event),
    )

    assert events
    first = events[0]
    assert isinstance(first["action"], str)
    assert first["action_reason"] == "random_policy_sample"


def test_run_random_policy_before_step_callback_reports_pending_action() -> None:
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    events: list[dict[str, object]] = []

    _ = run_random_policy(
        env=env,
        episodes=1,
        max_steps_per_episode=1,
        seed=7,
        before_step_callback=lambda event: events.append(event),
    )

    assert events
    first = events[0]
    assert isinstance(first["action"], str)
    assert first["action_reason"] == "random_policy_sample"
    assert isinstance(first["total_reward"], float)


def test_run_random_policy_uses_supplied_action_subset() -> None:
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=QueueStateProvider([_snapshot(failed=False)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "cancel"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    run_random_policy(env=env, episodes=1, max_steps_per_episode=3, seed=9, actions=("move_up",))

    assert actions.actions == ["move_up", "move_up", "move_up"]


def test_game_env_available_actions_filters_edges_and_walls() -> None:
    map_state = MapState(
        status="ok",
        width=3,
        height=3,
        player_position=GridPosition(0, 0),
        cells=(
            MapCellState(
                position=GridPosition(1, 0),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                is_wall=True,
            ),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([_snapshot(map_state=map_state)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "move_left", "move_right", "move_down", "wait"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    assert env.available_actions() == ("move_up",)


def test_run_random_policy_avoids_blocked_edge_moves() -> None:
    map_state = MapState(
        status="ok",
        width=2,
        height=1,
        player_position=GridPosition(0, 0),
    )
    actions = FakeActionAPI()
    env = GameEnv(
        action_api=actions,
        state_provider=QueueStateProvider([_snapshot(map_state=map_state)]),
        reset_strategy=NoopResetManager(),
        action_space=("move_left", "move_right"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )

    run_random_policy(env=env, episodes=1, max_steps_per_episode=4, seed=3)

    assert actions.actions == ["move_right", "move_right", "move_right", "move_right"]


def test_available_actions_hides_confirm_when_not_failed_and_shows_on_fail() -> None:
    non_fail_state = _snapshot(
        failed=False,
        map_state=MapState(status="ok", width=2, height=2, player_position=GridPosition(0, 0)),
    )
    fail_state = _snapshot(
        failed=True,
        map_state=MapState(status="ok", width=2, height=2, player_position=GridPosition(0, 0)),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([non_fail_state, fail_state]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "confirm"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()
    assert env.available_actions() == ("move_up",)

    env.step("move_up")
    assert env.available_actions() == ("move_up", "confirm")


def test_available_actions_adds_space_only_when_player_can_siphon() -> None:
    no_siphon = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=2,
            player_position=GridPosition(0, 0),
            siphons=(),
        )
    )
    with_siphon = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=2,
            player_position=GridPosition(1, 1),
            siphons=(GridPosition(1, 1),),
        )
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([no_siphon, with_siphon]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "space"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()
    assert env.available_actions() == ("move_up",)

    env.step("move_up")
    assert env.available_actions() == ("space",)


def test_available_actions_allows_space_from_ui_flag_even_when_not_on_tile() -> None:
    state = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=2,
            player_position=GridPosition(0, 0),
            siphons=(),
        ),
        can_siphon_now=True,
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([state]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "space"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    assert env.available_actions() == ("move_up", "space")


def test_available_actions_includes_only_owned_prog_slots() -> None:
    state = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=2,
            player_position=GridPosition(0, 0),
        ),
        inventory_state=InventoryState(
            status="ok",
            raw_prog_ids=(2, 7),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([state]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "prog_slot_1", "prog_slot_2", "prog_slot_3"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    assert env.available_actions() == ("move_up", "prog_slot_1", "prog_slot_2")


def test_available_actions_uses_prog_slots_ui_mask_when_present() -> None:
    state = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=2,
            player_position=GridPosition(0, 0),
        ),
        inventory_state=InventoryState(
            status="ok",
            raw_prog_ids=(2, 7, 9),
        ),
        prog_slots_available_mask=0b101,
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([state]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "prog_slot_1", "prog_slot_2", "prog_slot_3"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    assert env.available_actions() == ("move_up", "prog_slot_1", "prog_slot_3")


def test_available_actions_hides_prog_slots_when_inventory_unavailable() -> None:
    map_state = MapState(
        status="ok",
        width=2,
        height=2,
        player_position=GridPosition(0, 0),
    )
    missing_inventory = _snapshot(
        map_state=map_state,
        inventory_state=InventoryState(status="missing"),
    )
    invalid_inventory = _snapshot(
        map_state=map_state,
        inventory_state=InventoryState(status="invalid", error_code="read_failed"),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([missing_inventory, invalid_inventory]),
        reset_strategy=NoopResetManager(),
        action_space=("move_up", "prog_slot_1"),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()
    assert env.available_actions() == ("move_up",)

    env.step("move_up")
    assert env.available_actions() == ("move_up",)


def test_game_env_marks_premature_exit_attempt_in_step_info() -> None:
    before = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
            siphons=(GridPosition(0, 0),),
        ),
    )
    after = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(1, 0),
            siphons=(GridPosition(0, 0),),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([before, after]),
        reset_strategy=NoopResetManager(),
        action_space=("move_right",),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    _state, _reward, _done, info = env.step("move_right")

    assert info["premature_exit_attempt"] is True
    assert info["action_effective"] is True
    assert info["invalid_action_reason"] is None


def test_game_env_premature_exit_counts_type_zero_enemy_as_remaining() -> None:
    before = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
            siphons=(),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=0,
                    position=GridPosition(0, 0),
                    hp=3,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )
    after = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(1, 0),
            siphons=(),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=0,
                    position=GridPosition(0, 0),
                    hp=3,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([before, after]),
        reset_strategy=NoopResetManager(),
        action_space=("move_right",),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    _state, _reward, _done, info = env.step("move_right")

    assert info["premature_exit_attempt"] is True


def test_game_env_prog_action_backoff_after_ineffective_attempt() -> None:
    stale = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
        ),
        inventory_state=InventoryState(status="ok", raw_prog_ids=(2,)),
        energy=10,
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([stale, stale, stale]),
        reset_strategy=NoopResetManager(),
        action_space=("prog_slot_1", "move_right"),
        config=GameEnvConfig(require_non_terminal_on_reset=False, prog_slot_backoff_steps=3),
    )
    env.reset()

    _state, _reward, _done, info = env.step("prog_slot_1")

    assert info["action_effective"] is False
    assert info["invalid_action_reason"] == "prog_no_effect"
    assert "prog_slot_1" not in env.available_actions()


def test_game_env_space_action_is_effective_when_enemy_hp_changes_without_player_movement() -> None:
    before = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            enemies=(
                EnemyState(
                    slot=0,
                    type_id=1,
                    position=GridPosition(1, 0),
                    hp=5,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )
    after = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            enemies=(
                EnemyState(
                    slot=0,
                    type_id=1,
                    position=GridPosition(1, 0),
                    hp=4,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([before, after]),
        reset_strategy=NoopResetManager(),
        action_space=("space",),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    _state, _reward, _done, info = env.step("space")

    assert info["action_effective"] is True
    assert info["invalid_action_reason"] is None


def test_game_env_prog_action_is_effective_when_map_cell_state_changes_without_player_movement() -> None:
    before = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            cells=(
                MapCellState(
                    position=GridPosition(0, 0),
                    cell_type=2,
                    tile_variant=1,
                    wall_state=0,
                    points=3,
                ),
            ),
        ),
        inventory_state=InventoryState(status="ok", raw_prog_ids=(2,)),
        energy=10,
    )
    after = _snapshot(
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            cells=(
                MapCellState(
                    position=GridPosition(0, 0),
                    cell_type=2,
                    tile_variant=1,
                    wall_state=0,
                    points=2,
                ),
            ),
        ),
        inventory_state=InventoryState(status="ok", raw_prog_ids=(2,)),
        energy=10,
    )
    env = GameEnv(
        action_api=FakeActionAPI(),
        state_provider=QueueStateProvider([before, after]),
        reset_strategy=NoopResetManager(),
        action_space=("prog_slot_1",),
        config=GameEnvConfig(require_non_terminal_on_reset=False),
    )
    env.reset()

    _state, _reward, _done, info = env.step("prog_slot_1")

    assert info["action_effective"] is True
    assert info["invalid_action_reason"] is None
