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
        source_dir = Path(run_directory)
        if not source_dir.exists():
            raise FileNotFoundError(f"Hybrid checkpoint directory not found: {source_dir}")
        meta_path = source_dir / cls.META_FILE
        threat_path = source_dir / cls.THREAT_FILE
        hybrid_config_path = source_dir / cls.HYBRID_CONFIG_FILE
        training_state_path = source_dir / cls.TRAINING_STATE_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing hybrid checkpoint file: {meta_path}")
        if not threat_path.exists():
            raise FileNotFoundError(f"Missing hybrid checkpoint file: {threat_path}")
        if not hybrid_config_path.exists():
            raise FileNotFoundError(f"Missing hybrid checkpoint file: {hybrid_config_path}")
        if not training_state_path.exists():
            raise FileNotFoundError(f"Missing hybrid checkpoint file: {training_state_path}")

        meta_controller = MetaControllerDQN.load(meta_path)
        threat_controller = ThreatControllerDRQN.load(threat_path)
        hybrid_config = json.loads(hybrid_config_path.read_text(encoding="utf-8"))
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        if not isinstance(hybrid_config, dict):
            raise ValueError("hybrid_config.json must contain an object.")
        if not isinstance(training_state, dict):
            raise ValueError("training_state.json must contain an object.")
        return (meta_controller, threat_controller, hybrid_config, training_state)

