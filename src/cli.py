"""Master CLI for running 868-train tools through one command surface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Callable

from src.app import main as app_main
from src.config.fingerprint import main as fingerprint_main
from src.env.dqn_policy_runner import main as dqn_runner_main
from src.env.heuristic_policy_runner import main as heuristic_runner_main
from src.env.random_policy_runner import main as random_runner_main
from src.gui.dqn_runner_gui import main as dqn_gui_main
from src.memory.offset_smoke_test import main as offset_smoke_main
from src.memory.state_monitor_tui import main as state_monitor_main
from src.training.evaluate import main as evaluate_main

CommandHandler = Callable[[], None]


def _run_with_passthrough(main_fn: CommandHandler, passthrough_args: Sequence[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *passthrough_args]
        main_fn()
    finally:
        sys.argv = original_argv


def _add_passthrough_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    name: str,
    summary: str,
    details: str,
) -> None:
    subparsers.add_parser(name, help=summary, description=details)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="868-train",
        description=(
            "Master CLI for 868-train. Use one of the subcommands below to run runtime checks, "
            "monitoring tools, policy runners, or evaluation harnesses."
        ),
        epilog=(
            "To show a sub-tool's native options/help, forward args after '--'. "
            "Example: 'train-868 run-heuristic -- --help'."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_passthrough_parser(
        subparsers,
        name="bootstrap",
        summary="Run startup validation checks.",
        details="Runs src.app startup checks (binary fingerprint + offsets registry validation).",
    )
    _add_passthrough_parser(
        subparsers,
        name="fingerprint",
        summary="Binary fingerprint helper.",
        details="Runs src.config.fingerprint utility, e.g. --print-sha256 <path>.",
    )
    _add_passthrough_parser(
        subparsers,
        name="offset-smoke",
        summary="Live offsets smoke test.",
        details="Runs src.memory.offset_smoke_test against the live process.",
    )
    _add_passthrough_parser(
        subparsers,
        name="state-monitor",
        summary="Interactive memory monitor TUI.",
        details="Runs src.memory.state_monitor_tui for live field polling and passthrough controls.",
    )
    _add_passthrough_parser(
        subparsers,
        name="run-random",
        summary="Run random baseline episodes.",
        details="Runs src.env.random_policy_runner against the live game environment.",
    )
    _add_passthrough_parser(
        subparsers,
        name="run-heuristic",
        summary="Run heuristic baseline episodes.",
        details="Runs src.env.heuristic_policy_runner against the live game environment.",
    )
    _add_passthrough_parser(
        subparsers,
        name="run-dqn",
        summary="Run DQN train/eval episodes.",
        details="Runs src.env.dqn_policy_runner against the live game environment.",
    )
    _add_passthrough_parser(
        subparsers,
        name="dqn-gui",
        summary="Launch GUI for DQN run/eval/compare workflows.",
        details=(
            "Runs src.gui.dqn_runner_gui and exposes DQN runner/evaluation flags in an "
            "interactive window."
        ),
    )
    _add_passthrough_parser(
        subparsers,
        name="evaluate",
        summary="DQN checkpoint KPI evaluation.",
        details=(
            "Runs src.training.evaluate for fixed-seed KPI reports and "
            "checkpoint-vs-checkpoint comparisons."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    passthrough = list(unknown_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    handlers: dict[str, CommandHandler] = {
        "bootstrap": app_main,
        "fingerprint": fingerprint_main,
        "offset-smoke": offset_smoke_main,
        "state-monitor": state_monitor_main,
        "run-random": random_runner_main,
        "run-heuristic": heuristic_runner_main,
        "run-dqn": dqn_runner_main,
        "dqn-gui": dqn_gui_main,
        "evaluate": evaluate_main,
    }

    selected = handlers.get(args.command)
    if selected is None:
        parser.error(f"Unknown command: {args.command}")
        return
    _run_with_passthrough(selected, passthrough)


if __name__ == "__main__":
    main()
