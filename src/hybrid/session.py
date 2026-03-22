"""Hybrid runtime session orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.env.runner_common import (
    default_game_save_target_path,
    resolve_restore_save_source_path,
    restore_selected_save_file,
)
from src.env.runner_monitor import RunnerMonitorSession
from src.hybrid.astar_controller import AStarMovementController
from src.hybrid.checkpoint import HybridCheckpointManager
from src.hybrid.cli import (
    _build_hybrid_env,
    _build_meta_config,
    _build_meta_reward_weights,
    _build_parser,
    _build_threat_config,
    _build_threat_reward_weights,
    _next_run_directory,
    _validate_args,
)
from src.hybrid.coordinator import HybridCoordinator, HybridCoordinatorConfig
from src.hybrid.env import HybridLiveEnv
from src.hybrid.meta_controller import MetaControllerDQN
from src.hybrid.rewards import HybridMetaRewardWeights, HybridRewardSuite, HybridThreatRewardWeights
from src.hybrid.rollout import (
    _build_training_state_payload,
    _print_results,
    _run_rollouts,
)
from src.hybrid.threat_controller import ThreatControllerDRQN

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VICTORY_MONITOR_OUTPUT_FILE = "victory-transition-events.jsonl"
_VICTORY_MONITOR_LOG_FILE = "victory-transition-monitor.log"

class _VictoryMonitorProcess:
    """Manage the debugger-backed victory monitor as a child process."""

    def __init__(
        self,
        *,
        run_directory: Path,
        startup_delay_seconds: float = 0.35,
    ) -> None:
        self.output_path = Path(run_directory) / _VICTORY_MONITOR_OUTPUT_FILE
        self.log_path = Path(run_directory) / _VICTORY_MONITOR_LOG_FILE
        self._startup_delay_seconds = max(float(startup_delay_seconds), 0.0)
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._attached_pid: int | None = None
        self._read_offset = 0
        self._pending_fragment = ""

    def update_pid(self, pid: int) -> None:
        resolved_pid = int(pid)
        if self._attached_pid == resolved_pid and self._process is not None and self._process.poll() is None:
            return
        self._stop_current()
        self._start_for_pid(resolved_pid)

    def close(self) -> None:
        self._stop_current()

    def start_episode(self) -> None:
        self.consume_new_events()

    def consume_new_events(self) -> tuple[dict[str, Any], ...]:
        if not self.output_path.exists():
            return ()
        with self.output_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._read_offset)
            chunk = handle.read()
            self._read_offset = handle.tell()
        if not chunk:
            return ()
        buffered = self._pending_fragment + chunk
        lines = buffered.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            self._pending_fragment = lines.pop()
        else:
            self._pending_fragment = ""
        parsed: list[dict[str, Any]] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                parsed.append(payload)
        return tuple(parsed)

    def _start_for_pid(self, pid: int) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            "-m",
            "src.memory.victory_transition_monitor",
            "--pid",
            str(int(pid)),
            "--output",
            str(self.output_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(_REPO_ROOT),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError:
            self._close_log_handle()
            raise
        self._attached_pid = int(pid)
        if self._startup_delay_seconds > 0:
            time.sleep(self._startup_delay_seconds)
        if self._process.poll() is not None:
            exit_code = int(self._process.returncode or 0)
            log_tail = self._read_log_tail()
            self._stop_current()
            detail = f" Monitor log tail:\n{log_tail}" if log_tail else ""
            raise RuntimeError(
                "Victory transition monitor exited during startup "
                f"for pid={pid} with code={exit_code}.{detail}"
            )

    def _stop_current(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            self._process = None
        self._attached_pid = None
        self._close_log_handle()

    def _close_log_handle(self) -> None:
        if self._log_handle is None:
            return
        try:
            self._log_handle.close()
        finally:
            self._log_handle = None

    def _read_log_tail(self, *, max_lines: int = 12) -> str:
        if not self.log_path.exists():
            return ""
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])


def _hook_snapshot_value(event: dict[str, Any], field_name: str) -> Any | None:
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    field = snapshot.get(field_name)
    if not isinstance(field, dict) or field.get("status") != "ok":
        return None
    return field.get("value")


def _hook_snapshot_int(event: dict[str, Any], field_name: str) -> int | None:
    value = _hook_snapshot_value(event, field_name)
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _hook_snapshot_bool(event: dict[str, Any], field_name: str) -> bool | None:
    value = _hook_snapshot_value(event, field_name)
    if isinstance(value, bool):
        return value
    try:
        if value is None:
            return None
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _hook_event_is_victory_signal(event: dict[str, Any]) -> bool:
    return str(event.get("target_name") or "").strip() in {
        "normal_victory_flag_set",
        "points_victory_flag_set",
    }

def _build_hybrid_config_payload(
    args: argparse.Namespace,
    *,
    command: str,
    restore_save_source: Path | None,
    meta_reward_weights: HybridMetaRewardWeights,
    threat_reward_weights: HybridThreatRewardWeights,
    victory_monitor_enabled: bool,
    victory_monitor_output_path: Path | None,
    victory_monitor_log_path: Path | None,
) -> dict[str, Any]:
    return {
        "command": command,
        "run_tag": str(args.run_tag),
        "victory_monitor_enabled": bool(victory_monitor_enabled),
        "victory_monitor_output_path": (
            str(victory_monitor_output_path)
            if victory_monitor_output_path is not None
            else None
        ),
        "victory_monitor_log_path": (
            str(victory_monitor_log_path)
            if victory_monitor_log_path is not None
            else None
        ),
        "no_enemies_mode": bool(args.no_enemies),
        "movement_keys": str(args.movement_keys),
        "siphon_key": str(args.siphon_key),
        "prog_actions": bool(args.prog_actions),
        "restore_save_file": (
            str(restore_save_source)
            if restore_save_source is not None
            else None
        ),
        "restore_save_delay": float(args.restore_save_delay),
        "post_action_delay": float(args.post_action_delay),
        "action_ack_timeout": float(args.action_ack_timeout),
        "action_ack_poll_interval": float(args.action_ack_poll_interval),
        "post_action_delay_backoff": float(args.post_action_delay_backoff),
        "action_ack_timeout_backoff": float(args.action_ack_timeout_backoff),
        "action_ack_backoff_max_level": int(args.action_ack_backoff_max_level),
        "threat_trigger_distance": int(args.threat_trigger_distance),
        "phase_lock_min_steps": int(args.phase_lock_min_steps),
        "target_stall_release_steps": int(args.target_stall_release_steps),
        "warmstart_checkpoint": (
            str(args.warmstart_checkpoint)
            if getattr(args, "warmstart_checkpoint", None)
            else None
        ),
        "resume_checkpoint": (
            str(args.resume_checkpoint)
            if getattr(args, "resume_checkpoint", None)
            else None
        ),
        "meta_phase_override_credit_mode": str(getattr(args, "meta_phase_override_credit_mode", "skip_overridden")),
        "meta_reward_weights": asdict(meta_reward_weights),
        "threat_reward_weights": asdict(threat_reward_weights),
        "meta_config": vars(_build_meta_config(args)),
        "threat_config": vars(_build_threat_config(args)),
    }

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    command = str(args.command)
    training_commands = {"train-meta-no-enemies", "train-full-hierarchical"}
    victory_monitor_enabled = bool(args.victory_monitor) and command in training_commands
    run_directory: Path | None = None
    if command in training_commands:
        checkpoint_root = Path(str(args.checkpoint_root))
        run_directory = _next_run_directory(root=checkpoint_root, tag=str(args.run_tag))
        if victory_monitor_enabled:
            run_directory.mkdir(parents=True, exist_ok=True)

    monitor_enabled = bool(args.external_status_file)
    effective_window_input = bool(args.window_input)
    if bool(args.no_enemies):
        print("no_enemies_mode_enabled\tenemy entities will be suppressed.")
    restore_save_source = resolve_restore_save_source_path(args)
    restore_save_delay_seconds = max(float(args.restore_save_delay), 0.0)
    restore_save_target = (
        default_game_save_target_path()
        if restore_save_source is not None
        else None
    )
    if restore_save_source is not None and restore_save_target is not None:
        print(
            "savegame_restore_enabled\tsource={source}\ttarget={target}\tdelay_seconds={delay:.3f}".format(
                source=restore_save_source,
                target=restore_save_target,
                delay=restore_save_delay_seconds,
            )
        )

    def _restore_save_before_reset() -> None:
        if restore_save_source is None or restore_save_target is None:
            return
        restore_selected_save_file(
            source_path=restore_save_source,
            target_path=restore_save_target,
        )
        print(
            "savegame_restored_before_reset\tsource={source}\ttarget={target}\tdelay_seconds={delay:.3f}".format(
                source=restore_save_source,
                target=restore_save_target,
                delay=restore_save_delay_seconds,
            )
        )
        if restore_save_delay_seconds > 0:
            time.sleep(restore_save_delay_seconds)

    env: HybridLiveEnv | None = None
    victory_monitor_session: _VictoryMonitorProcess | None = None
    monitor_session = RunnerMonitorSession(
        executable_name=str(args.exe),
        runner_module="src.hybrid.runner",
        enabled=monitor_enabled,
        step_through=False,
        launch_monitor=False,
        external_status_file=(str(args.external_status_file) if args.external_status_file else None),
        external_control_file=(str(args.external_control_file) if args.external_control_file else None),
    )
    try:
        monitor_session.start()
        env = _build_hybrid_env(
            args,
            effective_window_input=effective_window_input,
            pre_reset_hook=(
                _restore_save_before_reset
                if restore_save_source is not None and restore_save_target is not None
                else None
            ),
        )
        if victory_monitor_enabled:
            assert run_directory is not None
            victory_monitor_session = _VictoryMonitorProcess(run_directory=run_directory)
            def _safe_update_victory_monitor_pid(pid: int) -> None:
                nonlocal victory_monitor_session
                if victory_monitor_session is None:
                    return
                try:
                    victory_monitor_session.update_pid(pid)
                except RuntimeError as error:
                    print(
                        "victory_transition_monitor_warning\tstartup_failed\tmessage={message}".format(
                            message=str(error).replace("\n", " | "),
                        )
                    )
                    victory_monitor_session.close()
                    victory_monitor_session = None

            env.add_runtime_binding_callback(_safe_update_victory_monitor_pid)
            if victory_monitor_session is not None:
                print(
                    "victory_transition_monitor_enabled\toutput={output}\tlog={log}".format(
                        output=victory_monitor_session.output_path,
                        log=victory_monitor_session.log_path,
                    )
                )

        meta_reward_weights = _build_meta_reward_weights(args)
        threat_reward_weights = _build_threat_reward_weights(args)
        reward_suite = HybridRewardSuite(
            meta_weights=meta_reward_weights,
            threat_weights=threat_reward_weights,
        )
        coordinator_config = HybridCoordinatorConfig(
            threat_trigger_distance=max(int(args.threat_trigger_distance), 1),
            exit_after_siphons_when_scripted=False,
            phase_lock_min_steps=max(int(args.phase_lock_min_steps), 0),
            target_stall_release_steps=max(int(args.target_stall_release_steps), 0),
        )
        coordinator = HybridCoordinator(
            meta_controller=MetaControllerDQN(config=_build_meta_config(args), seed=args.seed),
            threat_controller=ThreatControllerDRQN(config=_build_threat_config(args), seed=args.seed),
            movement_controller=AStarMovementController(),
            config=coordinator_config,
        )

        if command == "eval-hybrid":
            loaded_meta, loaded_threat, _bundle_config, _training_state = HybridCheckpointManager.load_bundle(
                run_directory=str(args.checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=loaded_threat,
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )
        elif getattr(args, "resume_checkpoint", None):
            loaded_meta, loaded_threat, _bundle_config, _training_state = HybridCheckpointManager.load_bundle(
                run_directory=str(args.resume_checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=loaded_threat,
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )
        elif command == "train-full-hierarchical" and getattr(args, "warmstart_checkpoint", None):
            loaded_meta, _bundle_config, _training_state = HybridCheckpointManager.load_warmstart_meta(
                run_directory=str(args.warmstart_checkpoint)
            )
            coordinator = HybridCoordinator(
                meta_controller=loaded_meta,
                threat_controller=ThreatControllerDRQN(config=_build_threat_config(args), seed=args.seed),
                movement_controller=AStarMovementController(),
                config=coordinator_config,
            )

        if command == "movement-test":
            train_meta = False
            train_threat = False
            use_meta = False
            use_threat = False
            explore_meta = False
            explore_threat = False
        elif command == "train-meta-no-enemies":
            train_meta = True
            train_threat = False
            use_meta = True
            use_threat = False
            explore_meta = True
            explore_threat = False
        elif command == "train-full-hierarchical":
            train_meta = False
            train_threat = True
            use_meta = True
            use_threat = True
            explore_meta = True
            explore_threat = True
        elif command == "eval-hybrid":
            train_meta = False
            train_threat = False
            use_meta = True
            use_threat = True
            explore_meta = False
            explore_threat = False
        else:  # pragma: no cover - argparse guards command values.
            raise ValueError(f"Unsupported command: {command}")

        results: list = []
        if command == "train-full-hierarchical":
            freeze_episodes = max(int(args.meta_freeze_episodes), 0)
            if freeze_episodes > 0:
                warmup_results = _run_rollouts(
                    env=env,
                    coordinator=coordinator,
                    reward_suite=reward_suite,
                    episodes=min(int(args.episodes), freeze_episodes),
                    max_steps=int(args.max_steps),
                    train_meta=False,
                    train_threat=True,
                    use_meta=True,
                    use_threat=True,
                    explore_meta=True,
                    explore_threat=True,
                    monitor_session=monitor_session,
                    monitor_enabled=monitor_enabled,
                    print_reward_breakdown=bool(args.print_reward_breakdown),
                    meta_phase_override_credit_mode=str(
                        getattr(args, "meta_phase_override_credit_mode", "skip_overridden")
                    ),
                    victory_monitor_session=victory_monitor_session,
                )
                results.extend(warmup_results)

            remaining = max(int(args.episodes) - len(results), 0)
            if remaining > 0:
                finetune_meta = bool(args.joint_finetune)
                finetune_results = _run_rollouts(
                    env=env,
                    coordinator=coordinator,
                    reward_suite=reward_suite,
                    episodes=remaining,
                    max_steps=int(args.max_steps),
                    train_meta=finetune_meta,
                    train_threat=True,
                    use_meta=True,
                    use_threat=True,
                    explore_meta=True,
                    explore_threat=True,
                    monitor_session=monitor_session,
                    monitor_enabled=monitor_enabled,
                    print_reward_breakdown=bool(args.print_reward_breakdown),
                    meta_phase_override_credit_mode=str(
                        getattr(args, "meta_phase_override_credit_mode", "skip_overridden")
                    ),
                    victory_monitor_session=victory_monitor_session,
                )
                results.extend(finetune_results)
        else:
            rollout_results = _run_rollouts(
                env=env,
                coordinator=coordinator,
                reward_suite=reward_suite,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                train_meta=train_meta,
                train_threat=train_threat,
                use_meta=use_meta,
                use_threat=use_threat,
                explore_meta=explore_meta,
                explore_threat=explore_threat,
                monitor_session=monitor_session,
                monitor_enabled=monitor_enabled,
                print_reward_breakdown=bool(args.print_reward_breakdown),
                meta_phase_override_credit_mode=str(
                    getattr(args, "meta_phase_override_credit_mode", "skip_overridden")
                ),
                victory_monitor_session=victory_monitor_session,
            )
            results.extend(rollout_results)

        finalized = tuple(results)
        _print_results(finalized)

        if command in {"train-meta-no-enemies", "train-full-hierarchical"}:
            assert run_directory is not None
            training_state_payload = _build_training_state_payload(
                results=finalized,
                episodes_requested=int(args.episodes),
                max_steps=int(args.max_steps),
            )
            bundle = HybridCheckpointManager.save_bundle(
                run_directory=run_directory,
                meta_controller=coordinator.meta_controller,
                threat_controller=coordinator.threat_controller,
                hybrid_config=_build_hybrid_config_payload(
                    args,
                    command=command,
                    restore_save_source=restore_save_source,
                    meta_reward_weights=meta_reward_weights,
                    threat_reward_weights=threat_reward_weights,
                    victory_monitor_enabled=victory_monitor_enabled,
                    victory_monitor_output_path=(
                        victory_monitor_session.output_path
                        if victory_monitor_session is not None
                        else None
                    ),
                    victory_monitor_log_path=(
                        victory_monitor_session.log_path
                        if victory_monitor_session is not None
                        else None
                    ),
                ),
                training_state=training_state_payload,
            )
            print(f"hybrid_checkpoint_saved\t{bundle.run_directory}")
            if command == "train-meta-no-enemies":
                best_pointer_path, best_run_directory = HybridCheckpointManager.update_best_meta_pointer(
                    run_directory=bundle.run_directory,
                    training_state=training_state_payload,
                    pointer_path=HybridCheckpointManager.default_meta_best_pointer_path(
                        checkpoint_root=Path(str(args.checkpoint_root))
                    ),
                )
                print(
                    "hybrid_meta_best\tpointer={pointer}\ttarget={target}".format(
                        pointer=best_pointer_path,
                        target=best_run_directory,
                    )
                )
    finally:
        if victory_monitor_session is not None:
            victory_monitor_session.close()
        if env is not None:
            env.close()
        monitor_session.close()
