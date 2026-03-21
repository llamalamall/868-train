"""CLI runner for hybrid hierarchical training and evaluation workflows."""

from src.hybrid.cli import (
    _build_meta_reward_weights,
    _build_parser,
    _build_threat_reward_weights,
    _validate_args,
)
from src.hybrid.rollout import (
    HybridEpisodeSummary,
    _build_training_state_payload,
    _build_training_summary,
    _classify_terminal_reason,
    _format_monitor_action_line,
    _format_monitor_actions,
    _format_monitor_training_line,
    _run_rollouts,
)
from src.hybrid.session import _build_hybrid_config_payload, _hook_event_is_victory_signal, main

__all__ = [
    "HybridEpisodeSummary",
    "_build_hybrid_config_payload",
    "_build_meta_reward_weights",
    "_build_parser",
    "_build_threat_reward_weights",
    "_build_training_state_payload",
    "_build_training_summary",
    "_classify_terminal_reason",
    "_format_monitor_action_line",
    "_format_monitor_actions",
    "_format_monitor_training_line",
    "_hook_event_is_victory_signal",
    "_run_rollouts",
    "_validate_args",
    "main",
]

if __name__ == "__main__":
    main()
