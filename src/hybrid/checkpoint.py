"""Hybrid checkpoint bundle management."""

from __future__ import annotations

import json
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
        source_dir = Path(run_directory)
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
