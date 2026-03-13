"""Tests for hybrid threat-controller replay and recurrent behavior."""

from __future__ import annotations

from src.hybrid.threat_controller import ThreatControllerDRQN, ThreatDRQNConfig
from src.hybrid.types import ThreatOverride


def _controller(*, seed: int = 5) -> ThreatControllerDRQN:
    return ThreatControllerDRQN(
        config=ThreatDRQNConfig(
            feature_count=4,
            hidden_size=8,
            replay_capacity=32,
            min_replay_size=99,
            batch_size=2,
            target_sync_interval=4,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decay_steps=1,
            sequence_length=8,
        ),
        seed=seed,
    )


def test_threat_controller_advances_hidden_state_during_exploration() -> None:
    controller = _controller()
    controller.start_episode()

    assert controller._hidden_state is None

    selected, reason, q_value = controller.select_override(
        features=(1.0, 0.0, 0.0, 0.0),
        threat_active=True,
        allowed_overrides=(ThreatOverride.ROUTE_DEFAULT, ThreatOverride.EVADE),
        explore=True,
    )

    assert selected in {ThreatOverride.ROUTE_DEFAULT, ThreatOverride.EVADE}
    assert reason == "threat_epsilon_explore"
    assert q_value is None
    assert controller._hidden_state is not None


def test_threat_controller_samples_multi_step_sequence_fragments() -> None:
    controller = _controller(seed=9)
    controller.start_episode()
    for index in range(3):
        controller.observe(
            features=(float(index), 0.0, 0.0, 0.0),
            chosen_override=ThreatOverride.EVADE,
            reward=-0.1 * index,
            next_features=(float(index + 1), 0.0, 0.0, 0.0),
            done=False,
            next_allowed_overrides=(ThreatOverride.ROUTE_DEFAULT, ThreatOverride.EVADE),
        )

    fragments = [controller._sample_sequence_fragment() for _ in range(12)]

    assert any(len(fragment) > 1 for fragment in fragments)
    assert all(1 <= len(fragment) <= controller.config.sequence_length for fragment in fragments)
