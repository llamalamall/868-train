"""Tests for high-level action API behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.controller.action_api import ActionAPI, ActionConfig, ActionExecutionError, ActionTimings, run_smoke_test_sequence


@dataclass
class FakeInputDriver:
    """Fake input driver that records calls for assertions."""

    tap_calls: list[dict[str, object]] = field(default_factory=list)
    window_tap_calls: list[dict[str, object]] = field(default_factory=list)
    waits: list[float] = field(default_factory=list)

    def tap_key(
        self,
        key_code: int,
        *,
        retries: int,
        retry_delay_seconds: float,
        press_duration_seconds: float,
        verification_hook=None,
    ) -> None:
        self.tap_calls.append(
            {
                "key_code": key_code,
                "retries": retries,
                "retry_delay_seconds": retry_delay_seconds,
                "press_duration_seconds": press_duration_seconds,
                "verification_hook": verification_hook,
            }
        )
        if verification_hook is not None:
            verification_hook(key_code, 1)

    def tap_key_to_window(
        self,
        hwnd: int,
        key_code: int,
        *,
        retries: int,
        retry_delay_seconds: float,
        press_duration_seconds: float,
        verification_hook=None,
    ) -> None:
        self.window_tap_calls.append(
            {
                "hwnd": hwnd,
                "key_code": key_code,
                "retries": retries,
                "retry_delay_seconds": retry_delay_seconds,
                "press_duration_seconds": press_duration_seconds,
                "verification_hook": verification_hook,
            }
        )
        if verification_hook is not None:
            verification_hook(key_code, 1)

    def sleep_wait(self, duration_seconds: float) -> None:
        self.waits.append(duration_seconds)


def test_directional_action_uses_mapped_key_and_timing() -> None:
    fake_driver = FakeInputDriver()
    config = ActionConfig(
        action_key_bindings={"move_up": "UP", "confirm": "ENTER", "cancel": "ESCAPE"},
        key_codes={"UP": 0x26, "ENTER": 0x0D, "ESCAPE": 0x1B},
        timings=ActionTimings(
            press_duration_seconds=0.07,
            inter_action_delay_seconds=0.03,
            retry_delay_seconds=0.01,
            wait_action_seconds=0.2,
            max_retries=4,
        ),
    )
    api = ActionAPI(input_driver=fake_driver, config=config)
    api.move_up()

    assert len(fake_driver.tap_calls) == 1
    assert fake_driver.tap_calls[0]["key_code"] == 0x26
    assert fake_driver.tap_calls[0]["retries"] == 4
    assert fake_driver.tap_calls[0]["press_duration_seconds"] == 0.07
    assert fake_driver.waits == [0.03]


def test_wait_action_uses_wait_timing_without_key_tap() -> None:
    fake_driver = FakeInputDriver()
    config = ActionConfig(
        action_key_bindings={"move_up": "UP"},
        key_codes={"UP": 0x26},
        timings=ActionTimings(wait_action_seconds=0.5, inter_action_delay_seconds=0.1),
    )
    api = ActionAPI(input_driver=fake_driver, config=config)
    api.wait()
    assert fake_driver.tap_calls == []
    assert fake_driver.waits == [0.5, 0.1]


def test_action_verification_hook_is_called() -> None:
    fake_driver = FakeInputDriver()
    api = ActionAPI(input_driver=fake_driver)
    observed: list[tuple[str, int, int]] = []

    def verify(action_name: str, key_code: int, attempt: int) -> bool:
        observed.append((action_name, key_code, attempt))
        return True

    api.perform_action("confirm", verification_hook=verify)
    assert observed == [("confirm", 0x0D, 1)]


def test_unknown_action_raises() -> None:
    fake_driver = FakeInputDriver()
    api = ActionAPI(input_driver=fake_driver)
    with pytest.raises(ActionExecutionError, match="Unknown action"):
        api.perform_action("not_real")


def test_send_key_name_uses_direct_key_mapping() -> None:
    fake_driver = FakeInputDriver()
    config = ActionConfig(
        action_key_bindings={"move_up": "UP"},
        key_codes={"UP": 0x26, "SPACE": 0x20},
        timings=ActionTimings(inter_action_delay_seconds=0.02, max_retries=2),
    )
    api = ActionAPI(input_driver=fake_driver, config=config)
    api.send_key_name("space")

    assert len(fake_driver.tap_calls) == 1
    assert fake_driver.tap_calls[0]["key_code"] == 0x20
    assert fake_driver.tap_calls[0]["retries"] == 2
    assert fake_driver.waits == [0.02]


def test_send_key_name_raises_when_key_not_configured() -> None:
    fake_driver = FakeInputDriver()
    config = ActionConfig(action_key_bindings={"move_up": "UP"}, key_codes={"UP": 0x26})
    api = ActionAPI(input_driver=fake_driver, config=config)
    with pytest.raises(ActionExecutionError, match="No virtual-key code configured"):
        api.send_key_name("SPACE")


def test_send_key_name_to_window_uses_target_hwnd() -> None:
    fake_driver = FakeInputDriver()
    config = ActionConfig(
        action_key_bindings={"move_up": "UP"},
        key_codes={"UP": 0x26, "SPACE": 0x20},
        timings=ActionTimings(inter_action_delay_seconds=0.02, max_retries=2),
    )
    api = ActionAPI(input_driver=fake_driver, config=config)
    api.send_key_name_to_window("space", hwnd=321)

    assert len(fake_driver.window_tap_calls) == 1
    assert fake_driver.window_tap_calls[0]["hwnd"] == 321
    assert fake_driver.window_tap_calls[0]["key_code"] == 0x20
    assert fake_driver.window_tap_calls[0]["retries"] == 2
    assert fake_driver.waits == [0.02]


def test_send_key_name_to_window_rejects_invalid_hwnd() -> None:
    fake_driver = FakeInputDriver()
    api = ActionAPI(input_driver=fake_driver)
    with pytest.raises(ActionExecutionError, match="Window handle must be > 0"):
        api.send_key_name_to_window("SPACE", hwnd=0)


def test_smoke_sequence_runs_in_expected_order() -> None:
    fake_driver = FakeInputDriver()
    api = ActionAPI(input_driver=fake_driver)
    sequence = run_smoke_test_sequence(api)
    assert sequence == [
        "move_up",
        "move_right",
        "confirm",
        "wait",
        "cancel",
        "move_left",
        "move_down",
    ]
    # 6 key taps and 8 waits (wait action + inter-action delay after every action)
    assert len(fake_driver.tap_calls) == 6
    assert len(fake_driver.waits) == 8
