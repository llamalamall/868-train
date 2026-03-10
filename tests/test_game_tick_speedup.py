"""Tests for in-memory game tick speedup patching."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.env.game_tick_speedup import (
    BackgroundMotionDisablePatcher,
    GameTickSpeedupPatcher,
    IdleFrameDelayBypassPatcher,
    TileAnimationFreezePatcher,
)
from src.memory.reader import ReadFailure, ReadResult
from src.memory.writer import WriteFailure, WriteResult


@dataclass
class FakeProcessMemory:
    bytes_by_address: dict[int, int] = field(default_factory=dict)

    def read(self, address: int, size: int) -> bytes:
        return bytes(self.bytes_by_address.get(address + index, 0) for index in range(size))

    def write(self, address: int, data: bytes) -> None:
        for index, value in enumerate(data):
            self.bytes_by_address[address + index] = value


@dataclass
class FakeReader:
    memory: FakeProcessMemory
    fail_reads: set[int] = field(default_factory=set)

    def read_bytes(self, address: int, size: int) -> ReadResult[bytes]:
        if address in self.fail_reads:
            return ReadResult.fail(
                ReadFailure(
                    code="read_failed",
                    message="forced read failure",
                    address=address,
                    size=size,
                )
            )
        return ReadResult.ok(self.memory.read(address, size))


@dataclass
class FakeWriter:
    memory: FakeProcessMemory
    fail_writes: set[int] = field(default_factory=set)
    writes: list[tuple[int, bytes]] = field(default_factory=list)

    def write_bytes(self, address: int, data: bytes) -> WriteResult:
        self.writes.append((address, bytes(data)))
        if address in self.fail_writes:
            return WriteResult.fail(
                WriteFailure(
                    code="write_failed",
                    message="forced write failure",
                    address=address,
                    size=len(data),
                )
            )
        self.memory.write(address, data)
        return WriteResult.ok()


def _load_tick_instruction(memory: FakeProcessMemory, *, module_base: int) -> None:
    instruction = bytes.fromhex("83 05 0D DF 06 00 10")
    instruction_address = module_base + 0xB08C
    memory.write(instruction_address, instruction)


def _load_idle_delay_instruction(memory: FakeProcessMemory, *, module_base: int) -> None:
    instruction = bytes.fromhex("8B CE")
    instruction_address = module_base + 0xB0C6
    memory.write(instruction_address, instruction)


def _load_background_motion_flag(
    memory: FakeProcessMemory,
    *,
    module_base: int,
    value: int,
) -> None:
    flag_address = module_base + 0x6CB80
    memory.write(flag_address, int(value).to_bytes(4, byteorder="little", signed=False))


def _load_tile_animation_counter_increment_instruction(
    memory: FakeProcessMemory,
    *,
    module_base: int,
) -> None:
    instruction_address = module_base + 0xB07B
    memory.write(instruction_address, bytes.fromhex("FF 05 33 12 12 00"))


def test_game_tick_speedup_noops_when_target_is_default() -> None:
    reader_calls: list[int] = []
    writer_calls: list[int] = []

    class _UnexpectedReader:
        def read_bytes(self, address: int, size: int) -> ReadResult[bytes]:
            reader_calls.append(address)
            return ReadResult.fail(ReadFailure(code="unexpected", message="unexpected call"))

    class _UnexpectedWriter:
        def write_bytes(self, address: int, data: bytes) -> WriteResult:
            writer_calls.append(address)
            return WriteResult.fail(WriteFailure(code="unexpected", message="unexpected call"))

    patcher = GameTickSpeedupPatcher(
        game_tick_ms=16,
        reader_factory=lambda _handle: _UnexpectedReader(),
        writer_factory=lambda _handle: _UnexpectedWriter(),
    )

    assert patcher.apply(process_handle=1, module_base=0x140000000) is False
    assert patcher.restore(process_handle=1, module_base=0x140000000) is False
    assert reader_calls == []
    assert writer_calls == []


def test_game_tick_speedup_applies_and_restores_tick_immediate() -> None:
    memory = FakeProcessMemory()
    module_base = 0x140000000
    _load_tick_instruction(memory, module_base=module_base)
    writer = FakeWriter(memory=memory)

    patcher = GameTickSpeedupPatcher(
        game_tick_ms=8,
        reader_factory=lambda _handle: FakeReader(memory=memory),
        writer_factory=lambda _handle: writer,
    )

    applied = patcher.apply(process_handle=123, module_base=module_base)
    assert applied is True
    assert memory.read(module_base + 0xB08C + 6, 1) == b"\x08"

    restored = patcher.restore(process_handle=123, module_base=module_base)
    assert restored is True
    assert memory.read(module_base + 0xB08C + 6, 1) == b"\x10"
    assert writer.writes[0] == (module_base + 0xB08C + 6, b"\x08")
    assert writer.writes[1] == (module_base + 0xB08C + 6, b"\x10")


def test_game_tick_speedup_warns_and_continues_when_write_fails(caplog) -> None:
    memory = FakeProcessMemory()
    module_base = 0x140000000
    _load_tick_instruction(memory, module_base=module_base)
    failing_writer = FakeWriter(memory=memory, fail_writes={module_base + 0xB08C + 6})

    patcher = GameTickSpeedupPatcher(
        game_tick_ms=8,
        reader_factory=lambda _handle: FakeReader(memory=memory),
        writer_factory=lambda _handle: failing_writer,
    )

    with caplog.at_level("WARNING"):
        applied = patcher.apply(process_handle=123, module_base=module_base)

    assert applied is False
    assert "instruction_write_failed" in caplog.text
    assert memory.read(module_base + 0xB08C + 6, 1) == b"\x10"


def test_game_tick_speedup_reapplies_after_recovering_new_process_handle() -> None:
    module_base = 0x140000000
    memory_a = FakeProcessMemory()
    memory_b = FakeProcessMemory()
    _load_tick_instruction(memory_a, module_base=module_base)
    _load_tick_instruction(memory_b, module_base=module_base)

    memory_by_handle = {
        1001: memory_a,
        1002: memory_b,
    }

    patcher = GameTickSpeedupPatcher(
        game_tick_ms=8,
        reader_factory=lambda handle: FakeReader(memory=memory_by_handle[int(handle)]),
        writer_factory=lambda handle: FakeWriter(memory=memory_by_handle[int(handle)]),
    )

    assert patcher.apply(process_handle=1001, module_base=module_base) is True
    assert memory_a.read(module_base + 0xB08C + 6, 1) == b"\x08"

    assert patcher.apply(process_handle=1002, module_base=module_base) is True
    assert memory_b.read(module_base + 0xB08C + 6, 1) == b"\x08"


def test_idle_frame_delay_bypass_applies_and_restores_instruction_bytes() -> None:
    memory = FakeProcessMemory()
    module_base = 0x140000000
    _load_idle_delay_instruction(memory, module_base=module_base)
    writer = FakeWriter(memory=memory)

    patcher = IdleFrameDelayBypassPatcher(
        enabled=True,
        reader_factory=lambda _handle: FakeReader(memory=memory),
        writer_factory=lambda _handle: writer,
    )

    applied = patcher.apply(process_handle=123, module_base=module_base)
    assert applied is True
    assert memory.read(module_base + 0xB0C6, 2) == bytes.fromhex("31 C9")

    restored = patcher.restore(process_handle=123, module_base=module_base)
    assert restored is True
    assert memory.read(module_base + 0xB0C6, 2) == bytes.fromhex("8B CE")
    assert writer.writes[0] == (module_base + 0xB0C6, bytes.fromhex("31 C9"))
    assert writer.writes[1] == (module_base + 0xB0C6, bytes.fromhex("8B CE"))


def test_background_motion_disable_applies_and_restores_flag_value() -> None:
    memory = FakeProcessMemory()
    module_base = 0x140000000
    _load_background_motion_flag(memory, module_base=module_base, value=0)
    writer = FakeWriter(memory=memory)

    patcher = BackgroundMotionDisablePatcher(
        enabled=True,
        reader_factory=lambda _handle: FakeReader(memory=memory),
        writer_factory=lambda _handle: writer,
    )

    applied = patcher.apply(process_handle=123, module_base=module_base)
    assert applied is True
    assert memory.read(module_base + 0x6CB80, 4) == (1).to_bytes(4, byteorder="little", signed=False)

    restored = patcher.restore(process_handle=123, module_base=module_base)
    assert restored is True
    assert memory.read(module_base + 0x6CB80, 4) == (0).to_bytes(4, byteorder="little", signed=False)
    assert writer.writes[0] == (
        module_base + 0x6CB80,
        (1).to_bytes(4, byteorder="little", signed=False),
    )
    assert writer.writes[1] == (
        module_base + 0x6CB80,
        (0).to_bytes(4, byteorder="little", signed=False),
    )


def test_tile_animation_freeze_applies_and_restores_instruction_bytes() -> None:
    memory = FakeProcessMemory()
    module_base = 0x140000000
    _load_tile_animation_counter_increment_instruction(memory, module_base=module_base)
    writer = FakeWriter(memory=memory)

    patcher = TileAnimationFreezePatcher(
        enabled=True,
        reader_factory=lambda _handle: FakeReader(memory=memory),
        writer_factory=lambda _handle: writer,
    )

    applied = patcher.apply(process_handle=123, module_base=module_base)
    assert applied is True
    assert memory.read(module_base + 0xB07B, 6) == bytes.fromhex("90 90 90 90 90 90")

    restored = patcher.restore(process_handle=123, module_base=module_base)
    assert restored is True
    assert memory.read(module_base + 0xB07B, 6) == bytes.fromhex("FF 05 33 12 12 00")
    assert writer.writes[0] == (module_base + 0xB07B, bytes.fromhex("90 90 90 90 90 90"))
    assert writer.writes[1] == (module_base + 0xB07B, bytes.fromhex("FF 05 33 12 12 00"))
