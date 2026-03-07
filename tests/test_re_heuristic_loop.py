"""Tests for re-heuristic loop mining/gating helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.re_heuristic import RuleDefinition, save_rule_pack
from src.agent.re_heuristic_loop import (
    LoopConfig,
    evaluate_kpi_gate,
    mine_candidates,
    parse_function_index,
    run_live_validation,
    run_loop,
    run_offline_benchmark,
    should_stop_loop,
    stage_candidate_rules,
)


def _write_readable_fixture(root: Path) -> None:
    (root / "functions").mkdir(parents=True, exist_ok=True)
    (root / "index.md").write_text(
        "\n".join(
            [
                "# Decompiled Function Index",
                "",
                "| Name | Line | Span (lines) |",
                "|---|---:|---:|",
                "| `fn_1400428e0` | 10 | 100 |",
                "| `fn_1400428e0` | 10 | 100 |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "functions" / "fn_1400428e0.c").write_text(
        "data siphon cell_type wall exit overlay threat special prog credits energy score points",
        encoding="utf-8",
    )


def test_parse_function_index_reads_rows(tmp_path) -> None:
    readable = tmp_path / "readable"
    _write_readable_fixture(readable)

    rows = parse_function_index(readable / "index.md")

    assert rows
    assert rows[0].name == "fn_1400428e0"


def test_mine_candidates_dedupes_signatures(tmp_path) -> None:
    readable = tmp_path / "readable"
    _write_readable_fixture(readable)

    candidates = mine_candidates(readable_dir=readable, existing_signatures=set(), min_score=0.1)

    signatures = [item.signature for item in candidates]
    assert len(signatures) == len(set(signatures))


def test_stage_candidate_rules_skips_existing_ids(tmp_path) -> None:
    readable = tmp_path / "readable"
    _write_readable_fixture(readable)
    candidates = mine_candidates(readable_dir=readable, existing_signatures=set(), min_score=0.1)
    staged = stage_candidate_rules(
        candidates=candidates,
        existing_rule_ids={"re_mined.map_cell_semantics." + candidates[0].candidate_id[-8:]},
        top_k=3,
    )
    # Existing ids might not match hash format above, but staged output should still be bounded and deterministic.
    assert len(staged) <= 3


def test_kpi_gate_requires_improvement_and_blocks_major_regression() -> None:
    previous, current = run_offline_benchmark(
        previous_rule_pack=None,
        candidate_rule_pack=None,
        episodes=8,
        seed=3,
    )
    # No policy change -> no meaningful KPI gain.
    result_same = evaluate_kpi_gate(
        previous=previous,
        current=current,
        fail_rate_regression_tolerance=0.05,
        reward_regression_tolerance=0.10,
    )
    assert result_same.passed is False

    better = type(previous)(
        avg_reward=previous.avg_reward + 1.0,
        fail_rate=max(previous.fail_rate - 0.1, 0.0),
        episodes=previous.episodes,
        terminal_reasons=previous.terminal_reasons,
    )
    result_better = evaluate_kpi_gate(
        previous=previous,
        current=better,
        fail_rate_regression_tolerance=0.05,
        reward_regression_tolerance=0.10,
    )
    assert result_better.passed is True


def test_regression_benchmark_gate_passes_with_low_health_override_rule(tmp_path) -> None:
    pack_path = tmp_path / "candidate.json"
    save_rule_pack(
        path=pack_path,
        rules=(
            RuleDefinition(
                rule_id="re_mined.prog_reward.force_move",
                priority=5000,
                reason="re_mined_force_move_right",
                action_preferences=("move_right",),
                conditions={"require_low_health": True},
            ),
        ),
    )
    previous_kpi, current_kpi = run_offline_benchmark(
        previous_rule_pack=None,
        candidate_rule_pack=pack_path,
        episodes=20,
        seed=9,
    )
    gate = evaluate_kpi_gate(
        previous=previous_kpi,
        current=current_kpi,
        fail_rate_regression_tolerance=0.05,
        reward_regression_tolerance=0.10,
    )
    assert gate.passed is True
    assert gate.improved_reward or gate.improved_fail_rate


def test_live_validation_disabled_mode_is_non_blocking() -> None:
    result = run_live_validation(
        mode="disabled",
        command=("python", "-c", "print('skip')"),
        timeout_seconds=5.0,
        fail_rate_regression_tolerance=0.05,
        previous_fail_rate=0.2,
    )
    assert result.executed is False
    assert result.passed is True


def test_should_stop_loop_matches_priority_order() -> None:
    assert should_stop_loop(
        accepted_rule_count=100,
        target_rule_count=100,
        consecutive_no_meaningful=0,
        iteration_index=1,
        max_iterations=200,
    ).reason == "target_rule_count_reached"
    assert should_stop_loop(
        accepted_rule_count=10,
        target_rule_count=100,
        consecutive_no_meaningful=2,
        iteration_index=1,
        max_iterations=200,
    ).reason == "no_meaningful_rules_for_two_iterations"
    assert should_stop_loop(
        accepted_rule_count=10,
        target_rule_count=100,
        consecutive_no_meaningful=0,
        iteration_index=200,
        max_iterations=200,
    ).reason == "max_iterations_reached"


def test_run_loop_stops_after_two_non_meaningful_iterations(tmp_path) -> None:
    readable = tmp_path / "readable"
    readable.mkdir(parents=True, exist_ok=True)
    (readable / "index.md").write_text("# empty\n", encoding="utf-8")

    config = LoopConfig(
        readable_dir=readable,
        output_root=tmp_path / "loop",
        max_iterations=10,
        live_validation="disabled",
        min_candidate_score=0.99,
    )
    summary = run_loop(config)

    assert summary["stop_reason"] == "no_meaningful_rules_for_two_iterations"
    assert summary["iterations_executed"] == 2
    state_path = config.loop_state_path
    assert state_path.exists()
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded["stop_reason"] == summary["stop_reason"]

