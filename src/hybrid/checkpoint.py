"""Hybrid checkpoint bundle management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.hybrid.meta_controller import MetaControllerDQN
from src.hybrid.threat_controller import ThreatControllerDRQN


@dataclass(frozen=True)
class HybridCheckpointBundle:
    """Resolved paths for one persisted hybrid checkpoint bundle."""

    run_directory: Path
    meta_controller_path: Path
    threat_controller_path: Path
    hybrid_config_path: Path
    training_state_path: Path


class HybridCheckpointManager:
    """Save/load helper for the multi-file hybrid checkpoint format."""

    VERSION = 1
    HYBRID_ROOT = Path("artifacts/hybrid")
    META_ROOT = HYBRID_ROOT / "meta"
    FULL_ROOT = HYBRID_ROOT / "full"
    META_BEST_POINTER_FILE = "hybrid-meta-best"
    META_FILE = "meta_controller.pt"
    THREAT_FILE = "threat_drqn.pt"
    HYBRID_CONFIG_FILE = "hybrid_config.json"
    TRAINING_STATE_FILE = "training_state.json"
    WARMSTART_REQUIRED_FILES = (
        META_FILE,
        HYBRID_CONFIG_FILE,
        TRAINING_STATE_FILE,
    )
    BUNDLE_REQUIRED_FILES = (
        META_FILE,
        THREAT_FILE,
        HYBRID_CONFIG_FILE,
        TRAINING_STATE_FILE,
    )

    @classmethod
    def default_meta_checkpoint_root(cls) -> Path:
        return cls.META_ROOT

    @classmethod
    def default_full_checkpoint_root(cls) -> Path:
        return cls.FULL_ROOT

    @classmethod
    def default_meta_best_pointer_path(
        cls,
        *,
        checkpoint_root: str | Path | None = None,
    ) -> Path:
        root = Path(checkpoint_root) if checkpoint_root is not None else cls.default_meta_checkpoint_root()
        return root.parent / cls.META_BEST_POINTER_FILE

    @classmethod
    def resolve_checkpoint_reference(
        cls,
        *,
        run_directory: str | Path,
        label: str = "Hybrid checkpoint directory",
    ) -> Path:
        source_path = Path(run_directory)
        if not source_path.exists() or source_path.is_dir():
            return source_path
        if not source_path.is_file():
            return source_path
        try:
            reference_text = source_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise OSError(f"Failed reading {label} pointer file: {source_path}") from error
        if not reference_text:
            raise ValueError(f"{label} pointer file is empty: {source_path}")
        resolved_target = Path(reference_text)
        if not resolved_target.is_absolute():
            resolved_target = (source_path.parent / resolved_target).resolve()
        return resolved_target

    @classmethod
    def resolve_required_paths(
        cls,
        *,
        run_directory: str | Path,
    ) -> dict[str, Path]:
        source_dir = Path(run_directory)
        return {
            cls.META_FILE: source_dir / cls.META_FILE,
            cls.THREAT_FILE: source_dir / cls.THREAT_FILE,
            cls.HYBRID_CONFIG_FILE: source_dir / cls.HYBRID_CONFIG_FILE,
            cls.TRAINING_STATE_FILE: source_dir / cls.TRAINING_STATE_FILE,
        }

    @classmethod
    def validate_bundle_directory(
        cls,
        *,
        run_directory: str | Path,
        required_files: tuple[str, ...] | None = None,
        label: str = "Hybrid checkpoint directory",
    ) -> Path:
        source_dir = cls.resolve_checkpoint_reference(run_directory=run_directory, label=label)
        if not source_dir.exists():
            raise FileNotFoundError(f"{label} not found: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"{label} must be a directory: {source_dir}")
        required = required_files or cls.BUNDLE_REQUIRED_FILES
        resolved_paths = cls.resolve_required_paths(run_directory=source_dir)
        missing = tuple(name for name in required if not resolved_paths[name].exists())
        if missing:
            available_files = sorted(
                child.name
                for child in source_dir.iterdir()
                if child.is_file()
            )
            available_suffix = (
                f" Available files: {', '.join(available_files)}."
                if available_files
                else " Directory is empty."
            )
            raise FileNotFoundError(
                f"{label} is incomplete: {source_dir}. "
                f"Missing files: {', '.join(missing)}.{available_suffix}"
            )
        return source_dir

    @classmethod
    def save_bundle(
        cls,
        *,
        run_directory: str | Path,
        meta_controller: MetaControllerDQN,
        threat_controller: ThreatControllerDRQN,
        hybrid_config: dict[str, Any],
        training_state: dict[str, Any],
    ) -> HybridCheckpointBundle:
        target_dir = Path(run_directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_controller.save(target_dir / cls.META_FILE)
        threat_path = threat_controller.save(target_dir / cls.THREAT_FILE)
        hybrid_config_path = target_dir / cls.HYBRID_CONFIG_FILE
        training_state_path = target_dir / cls.TRAINING_STATE_FILE
        hybrid_payload = {
            "version": cls.VERSION,
            **dict(hybrid_config),
        }
        training_payload = {
            "version": cls.VERSION,
            **dict(training_state),
        }
        hybrid_config_path.write_text(json.dumps(hybrid_payload, indent=2), encoding="utf-8")
        training_state_path.write_text(json.dumps(training_payload, indent=2), encoding="utf-8")
        return HybridCheckpointBundle(
            run_directory=target_dir,
            meta_controller_path=meta_path,
            threat_controller_path=threat_path,
            hybrid_config_path=hybrid_config_path,
            training_state_path=training_state_path,
        )

    @classmethod
    def update_best_meta_pointer(
        cls,
        *,
        run_directory: str | Path,
        training_state: dict[str, Any],
        pointer_path: str | Path | None = None,
    ) -> tuple[Path, Path]:
        candidate_directory = Path(run_directory)
        resolved_pointer_path = (
            Path(pointer_path)
            if pointer_path is not None
            else cls.default_meta_best_pointer_path(checkpoint_root=candidate_directory.parent)
        )
        current_best_directory = cls._select_best_meta_directory(
            candidate_directory=candidate_directory,
            candidate_training_state=training_state,
            pointer_path=resolved_pointer_path,
        )
        resolved_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        relative_target = os.path.relpath(current_best_directory, start=resolved_pointer_path.parent)
        resolved_pointer_path.write_text(relative_target + "\n", encoding="utf-8")
        return resolved_pointer_path, current_best_directory

    @classmethod
    def load_bundle(
        cls,
        *,
        run_directory: str | Path,
    ) -> tuple[MetaControllerDQN, ThreatControllerDRQN, dict[str, Any], dict[str, Any]]:
        source_dir = cls.validate_bundle_directory(run_directory=run_directory)
        resolved_paths = cls.resolve_required_paths(run_directory=source_dir)
        meta_path = resolved_paths[cls.META_FILE]
        threat_path = resolved_paths[cls.THREAT_FILE]
        hybrid_config_path = resolved_paths[cls.HYBRID_CONFIG_FILE]
        training_state_path = resolved_paths[cls.TRAINING_STATE_FILE]

        meta_controller = MetaControllerDQN.load(meta_path)
        threat_controller = ThreatControllerDRQN.load(threat_path)
        hybrid_config = json.loads(hybrid_config_path.read_text(encoding="utf-8"))
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        if not isinstance(hybrid_config, dict):
            raise ValueError("hybrid_config.json must contain an object.")
        if not isinstance(training_state, dict):
            raise ValueError("training_state.json must contain an object.")
        return (meta_controller, threat_controller, hybrid_config, training_state)

    @classmethod
    def load_warmstart_meta(
        cls,
        *,
        run_directory: str | Path,
    ) -> tuple[MetaControllerDQN, dict[str, Any], dict[str, Any]]:
        source_dir = cls.validate_bundle_directory(
            run_directory=run_directory,
            required_files=cls.WARMSTART_REQUIRED_FILES,
            label="Hybrid warmstart checkpoint directory",
        )
        resolved_paths = cls.resolve_required_paths(run_directory=source_dir)
        meta_path = resolved_paths[cls.META_FILE]
        hybrid_config_path = resolved_paths[cls.HYBRID_CONFIG_FILE]
        training_state_path = resolved_paths[cls.TRAINING_STATE_FILE]

        meta_controller = MetaControllerDQN.load(meta_path)
        hybrid_config = json.loads(hybrid_config_path.read_text(encoding="utf-8"))
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        if not isinstance(hybrid_config, dict):
            raise ValueError("hybrid_config.json must contain an object.")
        if not isinstance(training_state, dict):
            raise ValueError("training_state.json must contain an object.")
        return (meta_controller, hybrid_config, training_state)

    @classmethod
    def _select_best_meta_directory(
        cls,
        *,
        candidate_directory: Path,
        candidate_training_state: dict[str, Any],
        pointer_path: Path,
    ) -> Path:
        candidate_key = cls._meta_training_rank(
            training_state=candidate_training_state,
            run_directory=candidate_directory,
        )
        existing_directory, existing_training_state = cls._load_best_meta_candidate(pointer_path=pointer_path)
        if existing_directory is None or existing_training_state is None:
            return candidate_directory
        existing_key = cls._meta_training_rank(
            training_state=existing_training_state,
            run_directory=existing_directory,
        )
        if candidate_key >= existing_key:
            return candidate_directory
        return existing_directory

    @classmethod
    def _load_best_meta_candidate(
        cls,
        *,
        pointer_path: Path,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        if not pointer_path.exists():
            return (None, None)
        try:
            existing_directory = cls.resolve_checkpoint_reference(
                run_directory=pointer_path,
                label="Hybrid meta best pointer",
            )
            resolved_paths = cls.resolve_required_paths(run_directory=existing_directory)
            training_state = json.loads(
                resolved_paths[cls.TRAINING_STATE_FILE].read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError, NotADirectoryError):
            return (None, None)
        if not isinstance(training_state, dict):
            return (None, None)
        return (existing_directory, training_state)

    @classmethod
    def _meta_training_rank(
        cls,
        *,
        training_state: dict[str, Any],
        run_directory: Path,
    ) -> tuple[float, float, float, float, int]:
        summary = training_state.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        non_death_terminal_rate = cls._coerce_float(summary.get("non_death_terminal_rate"))
        avg_total_reward = cls._coerce_float(summary.get("avg_total_reward"))
        done_rate = cls._coerce_float(summary.get("done_rate"))
        unexpected_start_screen_rate = cls._coerce_float(summary.get("unexpected_start_screen_rate"))
        return (
            non_death_terminal_rate,
            avg_total_reward,
            done_rate,
            -unexpected_start_screen_rate,
            cls._run_directory_sort_key(run_directory),
        )

    @staticmethod
    def _coerce_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _run_directory_sort_key(run_directory: Path) -> int:
        digits = "".join(character for character in run_directory.name if character.isdigit())
        if not digits:
            return 0
        try:
            return int(digits)
        except ValueError:
            return 0
