"""Tests for master CLI command dispatch."""

from __future__ import annotations

import sys

from src import cli


def _install_stub(monkeypatch, module_path: str, captured: list[tuple[str, list[str]]]) -> None:
    command_name = module_path.split(".")[-1]

    def _stub() -> None:
        captured.append((command_name, list(sys.argv[1:])))

    monkeypatch.setattr(module_path, _stub)


def test_master_cli_dispatches_run_heuristic_with_passthrough(monkeypatch) -> None:
    captured: list[tuple[str, list[str]]] = []
    _install_stub(monkeypatch, "src.cli.heuristic_runner_main", captured)

    cli.main(["run-heuristic", "--episodes", "7", "--max-steps", "33"])

    assert captured == [("heuristic_runner_main", ["--episodes", "7", "--max-steps", "33"])]


def test_master_cli_strips_passthrough_separator(monkeypatch) -> None:
    captured: list[tuple[str, list[str]]] = []
    _install_stub(monkeypatch, "src.cli.random_runner_main", captured)

    cli.main(["run-random", "--", "--episodes", "5"])

    assert captured == [("random_runner_main", ["--episodes", "5"])]


def test_master_cli_dispatches_bootstrap(monkeypatch) -> None:
    captured: list[tuple[str, list[str]]] = []
    _install_stub(monkeypatch, "src.cli.app_main", captured)

    cli.main(["bootstrap"])

    assert captured == [("app_main", [])]
