"""Tests for TUI control key passthrough mapping."""

from __future__ import annotations

from src.state.schema import EnemyState, GridPosition, MapCellState, MapState, ResourceCellState
from src.memory.state_monitor_tui import (
    FieldSnapshot,
    PollSnapshot,
    format_collected_progs_status,
    increment_external_advance_counter,
    is_fail_state_detected,
    load_external_advance_counter,
    load_external_status_snapshot,
    map_tui_key_to_passthrough_key,
    render_ascii_map,
    summarize_board_state,
)


def test_passthrough_mapping_for_supported_keys() -> None:
    assert map_tui_key_to_passthrough_key("up") == "UP"
    assert map_tui_key_to_passthrough_key("right") == "RIGHT"
    assert map_tui_key_to_passthrough_key("escape") == "ESCAPE"
    assert map_tui_key_to_passthrough_key("space") == "SPACE"
    assert map_tui_key_to_passthrough_key("0") == "0"
    assert map_tui_key_to_passthrough_key("9") == "9"


def test_passthrough_mapping_rejects_unknown_keys() -> None:
    assert map_tui_key_to_passthrough_key("z") is None
    assert map_tui_key_to_passthrough_key("enter") is None


def _snapshot_with_health(value: str, status: str = "ok") -> PollSnapshot:
    return PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="player_health",
                data_type="int32",
                confidence="high",
                address="0x200000",
                value=value,
                status=status,
                error="",
            ),
        ),
    )


def test_is_fail_state_detected_true_when_health_is_negative_one() -> None:
    assert is_fail_state_detected(_snapshot_with_health("-1"))
    assert is_fail_state_detected(_snapshot_with_health("-1.0"))


def test_is_fail_state_detected_false_for_non_terminal_or_error_values() -> None:
    assert not is_fail_state_detected(_snapshot_with_health("0"))
    assert not is_fail_state_detected(_snapshot_with_health("5"))
    assert not is_fail_state_detected(_snapshot_with_health("-2"))
    assert not is_fail_state_detected(_snapshot_with_health("-1", status="error"))


def test_is_fail_state_detected_false_when_field_missing() -> None:
    snapshot = PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="player_health",
                data_type="int32",
                confidence="high",
                address="0x200004",
                value="10",
                status="ok",
                error="",
            ),
        ),
    )
    assert not is_fail_state_detected(snapshot)


def test_format_collected_progs_status_from_snapshot() -> None:
    snapshot = PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="collected_progs",
                data_type="array<int32>",
                confidence="medium",
                address="0x201000",
                value="count=3 [.wait(0), .pull(4), .hack(18)]",
                status="ok",
                error="",
            ),
        ),
    )
    status = format_collected_progs_status(snapshot)
    assert status == "progs=count=3 [.wait(0), .pull(4), .hack(18)]"


def test_format_collected_progs_status_handles_error_status() -> None:
    snapshot = PollSnapshot(
        timestamp="2026-03-03T00:00:00+00:00",
        fields=(
            FieldSnapshot(
                name="collected_progs",
                data_type="array<int32>",
                confidence="medium",
                address="N/A",
                value="",
                status="resolve_error",
                error="null_pointer",
            ),
        ),
    )
    assert format_collected_progs_status(snapshot) == "progs_status=resolve_error"


def test_summarize_board_state_includes_requested_aggregates() -> None:
    map_state = MapState(
        status="ok",
        cells=(
            MapCellState(
                position=GridPosition(x=0, y=0),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                prog_id=4,
                points=5,
                is_wall=True,
            ),
            MapCellState(
                position=GridPosition(x=0, y=1),
                cell_type=0,
                tile_variant=0,
                wall_state=0,
                prog_id=None,
                points=2,
            ),
            MapCellState(
                position=GridPosition(x=1, y=0),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                prog_id=7,
                points=0,
                is_wall=True,
            ),
        ),
        resource_cells=(
            ResourceCellState(position=GridPosition(x=0, y=1), credits=3, energy=2, points=2),
            ResourceCellState(position=GridPosition(x=2, y=2), credits=1, energy=5, points=0),
        ),
        enemies=(
            EnemyState(
                slot=0,
                type_id=2,
                position=GridPosition(x=1, y=1),
                hp=3,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=1,
                type_id=2,
                position=GridPosition(x=3, y=4),
                hp=4,
                state=1,
                in_bounds=True,
            ),
            EnemyState(
                slot=2,
                type_id=7,
                position=GridPosition(x=0, y=5),
                hp=1,
                state=0,
                in_bounds=True,
            ),
        ),
    )

    summary = summarize_board_state(map_state)
    assert summary == "board progs=2 points=7 credits=4 energy=7 enemies=virus(2):2,7:1"


def test_summarize_board_state_reports_non_ok_status() -> None:
    assert summarize_board_state(MapState(status="missing")) == "board_status=missing"


def test_summarize_board_state_uses_enemy_names_when_provided() -> None:
    map_state = MapState(
        status="ok",
        enemies=(
            EnemyState(
                slot=0,
                type_id=2,
                position=GridPosition(x=1, y=1),
                hp=3,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=1,
                type_id=2,
                position=GridPosition(x=2, y=1),
                hp=2,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=2,
                type_id=7,
                position=GridPosition(x=3, y=1),
                hp=1,
                state=0,
                in_bounds=True,
            ),
        ),
    )

    summary = summarize_board_state(map_state, enemy_name_by_id={2: "virus", 7: "daemon"})
    assert summary == "board progs=0 points=0 credits=0 energy=0 enemies=virus(2):2,daemon(7):1"


def test_summarize_board_state_uses_enemy_overrides_and_ignores_type_zero() -> None:
    map_state = MapState(
        status="ok",
        enemies=(
            EnemyState(
                slot=0,
                type_id=0,
                position=GridPosition(x=0, y=0),
                hp=1,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=1,
                type_id=2,
                position=GridPosition(x=1, y=0),
                hp=1,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=2,
                type_id=3,
                position=GridPosition(x=2, y=0),
                hp=1,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=3,
                type_id=4,
                position=GridPosition(x=3, y=0),
                hp=1,
                state=0,
                in_bounds=True,
            ),
            EnemyState(
                slot=4,
                type_id=5,
                position=GridPosition(x=4, y=0),
                hp=1,
                state=0,
                in_bounds=True,
            ),
        ),
    )

    summary = summarize_board_state(map_state, enemy_name_by_id={2: "wrong", 3: "wrong"})
    assert summary == (
        "board progs=0 points=0 credits=0 energy=0 "
        "enemies=virus(2):1,classic(3):1,glitch(4):1,cryptog(5):1"
    )


def test_render_ascii_map_shows_key_board_features() -> None:
    map_state = MapState(
        status="ok",
        width=3,
        height=2,
        cells=(
            MapCellState(
                position=GridPosition(x=0, y=0),
                cell_type=1,
                tile_variant=0,
                wall_state=0,
                is_wall=True,
            ),
            MapCellState(
                position=GridPosition(x=1, y=0),
                cell_type=0,
                tile_variant=0,
                wall_state=0,
                credits=2,
            ),
            MapCellState(
                position=GridPosition(x=2, y=0),
                cell_type=3,
                tile_variant=0,
                wall_state=0,
                is_exit=True,
            ),
            MapCellState(
                position=GridPosition(x=0, y=1),
                cell_type=0,
                tile_variant=0,
                wall_state=0,
            ),
            MapCellState(
                position=GridPosition(x=1, y=1),
                cell_type=0,
                tile_variant=0,
                wall_state=0,
            ),
            MapCellState(
                position=GridPosition(x=2, y=1),
                cell_type=0,
                tile_variant=0,
                wall_state=0,
            ),
        ),
        siphons=(GridPosition(x=1, y=1),),
        player_position=GridPosition(x=1, y=1),
        exit_position=GridPosition(x=2, y=0),
        enemies=(
            EnemyState(
                slot=0,
                type_id=0,
                position=GridPosition(x=1, y=1),
                hp=3,
                state=0,
                in_bounds=True,
            ),
        ),
    )

    rendered = render_ascii_map(map_state)
    assert "map  [bright_black]\u00b7[/] empty  [yellow]\u2593[/] wall" in rendered
    assert "[red]\u2620[/] enemy  [bright_white]\u263a[/] player" in rendered
    assert "1|[bright_black]\u00b7[/][bright_white]\u263a[/][bright_black]\u00b7[/]" in rendered
    assert "0|[yellow]\u2593[/][cyan]\u00a4[/][magenta]\u21e9[/]" in rendered
    assert rendered.index("1|[bright_black]\u00b7[/][bright_white]\u263a[/][bright_black]\u00b7[/]") < rendered.index(
        "0|[yellow]\u2593[/][cyan]\u00a4[/][magenta]\u21e9[/]"
    )
    assert "  012" in rendered


def test_render_ascii_map_reports_non_ok_status() -> None:
    assert render_ascii_map(MapState(status="missing")) == "map unavailable (missing)"


def test_load_external_status_snapshot_returns_empty_for_missing_file(tmp_path) -> None:
    status = load_external_status_snapshot(tmp_path / "missing.json")
    assert status.training_line == ""
    assert status.action_line == ""


def test_load_external_status_snapshot_reads_training_and_action_lines(tmp_path) -> None:
    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"training_line":"episode=1 step=4 total_reward=2.100","action_line":"action=move_up reason=collect_siphon"}',
        encoding="utf-8",
    )

    status = load_external_status_snapshot(status_file)
    assert status.training_line == "episode=1 step=4 total_reward=2.100"
    assert status.action_line == "action=move_up reason=collect_siphon"


def test_load_external_advance_counter_returns_zero_for_missing_file(tmp_path) -> None:
    control_file = tmp_path / "control.json"
    assert load_external_advance_counter(control_file) == 0


def test_increment_external_advance_counter_updates_counter_file(tmp_path) -> None:
    control_file = tmp_path / "control.json"

    assert increment_external_advance_counter(control_file) == 1
    assert increment_external_advance_counter(control_file) == 2
    assert load_external_advance_counter(control_file) == 2
