"""Feetech TTL bus + HD1910 servos + body IMU, behind RobotIO.

Register map: kissqy/Microduck-kissqy duck-control/src/{feetech,hd1910}.rs, verified on an
official-skeleton + 15x HD1910 robot (control.43). Only these registers are touched:

  40       torque enable (1 B)
  42..47   goal: position (2 B, sign-magnitude bit 15) + 4 zero bytes; 41 (acceleration)
           is never written
  48..49   torque limit (2 B) <- 1000
  50       RAM P gain (1 B)   <- 5 (what microduck_rl's HD1910 BAM fit is trained against)
  51       RAM D gain (1 B)   <- 0
  56..70   15-byte feedback: pos(56-57, s-m bit15) speed(58-59, s-m bit15, 50 steps/s/count)
           load(60-61) volt(62, 0.1 V) temp(63) ... moving(66) ... current(69-70)

Position: 4096 counts/turn; rad = (raw - zero_raw) * 2pi/4096 * direction (per-joint
calibration, see calibration.py). RAM P/D/limit are volatile: they are (re)written on
every torque-on.

IMU: the FT6 firmware on the RevA board (STM32G031 + LSM6DSV16X) answers as ID 200 on
this bus; its address-56 block starts with the same 12 bytes as the official imu_to_dxl
board (gyro + SFLP quaternion), decoded by imu.SflpDecoder. A different IMU can be
plugged in through ``imu_source``: any object with ``read() -> (ImuData, ready)`` and
``latest_block() -> bytes | None`` — e.g. ``esp32_link.Esp32Imu`` for an LSM6DSV16X
read by an ESP32 and streamed over a UART.

Half-duplex: the adapter must switch direction itself (Feetech URT-1, Waveshare bus
servo adapters, or a 74HC126-style circuit). Echo of the sent bytes is auto-detected on
the first ping (echo="auto"); force it with echo=True/False.
"""

from __future__ import annotations

import math
import struct
import time

import numpy as np

from .calibration import Calibration
from .contract import IMU_ID, MOUTH_INDEX, NUM_JOINTS
from .imu import BLOCK_LEN, ImuData, SflpDecoder
from .io import RobotIO, Sensors

BAUD = 1_000_000
PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83

REG_TORQUE = 40
REG_GOAL = 42
REG_TORQUE_LIMIT = 48
REG_RAM_P = 50
REG_RAM_D = 51
REG_FEEDBACK = 56
FEEDBACK_LEN = 15
SPEED_RAD_S_PER_COUNT = 50.0 * 2.0 * math.pi / 4096.0
FIRMWARE_P = 5
TORQUE_LIMIT = 1000


class BusError(RuntimeError):
    pass


def _packet(id_: int, instr: int, params: bytes = b"") -> bytes:
    body = bytes([id_, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def _signmag(raw: int, sign_bit: int = 15) -> int:
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


def _signmag_word(value: int) -> int:
    if not -32767 <= value <= 32767:
        raise BusError(f"position {value} outside signed 15-bit range")
    return abs(value) | (0x8000 if value < 0 else 0)


class FeetechBus:
    def __init__(self, port: str, baud: int = BAUD, timeout_s: float = 0.01, echo: bool | str = "auto"):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=timeout_s)
        try:  # ASYNC_LOW_LATENCY: no driver-side batching of RX bytes (FTDI/CH34x/CP210x)
            self.ser.set_low_latency_mode(True)
        except Exception:  # noqa: BLE001 - not supported by every driver/platform
            pass
        self.echo = echo
        self.bad_packets = 0
        self._rx = b""

    def close(self):
        self.ser.close()

    def _send(self, pkt: bytes):
        self.ser.reset_input_buffer()
        self._rx = b""
        self.ser.write(pkt)
        self.ser.flush()
        if self.echo == "auto":
            self._detect_echo(pkt)
        elif self.echo:
            self.ser.read(len(pkt))

    def _detect_echo(self, pkt: bytes) -> None:
        """First transaction: does the adapter hand our own bytes back?"""
        head = self.ser.read(len(pkt))
        if head == pkt:
            self.echo = True
        elif head:  # something else answered first: no echo
            self.echo = False
            self._rx = head
        # nothing at all (e.g. pinging an absent ID): stay "auto", decide next time

    def _status(self, deadline: float) -> tuple[int, int, bytes]:
        """Next status packet. Bytes after it stay in self._rx for the next call (a sync
        read's replies often arrive in one chunk)."""
        while True:
            buf = self._rx
            while len(buf) >= 2 and buf[:2] != b"\xff\xff":
                buf = buf[1:]
            if len(buf) >= 4 and buf[3] >= 2:
                n = buf[3] + 4
                if len(buf) >= n:
                    pkt, self._rx = buf[:n], buf[n:]
                    if (~sum(pkt[2:-1])) & 0xFF != pkt[-1]:
                        self.bad_packets += 1
                        continue
                    return pkt[2], pkt[4], pkt[5:-1]
            elif len(buf) >= 4:  # impossible length byte: resync
                buf = buf[1:]
            self._rx = buf
            if time.monotonic() >= deadline:
                raise BusError("status packet timeout")
            self._rx += self.ser.read(max(1, self.ser.in_waiting))

    def ping(self, id_: int, timeout_s: float = 0.02) -> bool:
        self._send(_packet(id_, PING))
        try:
            self._status(time.monotonic() + timeout_s)
            return True
        except BusError:
            return False

    def read(self, id_: int, addr: int, n: int, timeout_s: float = 0.02) -> bytes:
        self._send(_packet(id_, READ, bytes([addr, n])))
        rid, err, data = self._status(time.monotonic() + timeout_s)
        if rid != id_ or len(data) != n:
            raise BusError(f"read {id_}@{addr}: got id {rid}, {len(data)} B")
        return data

    def write(self, id_: int, addr: int, data: bytes, timeout_s: float = 0.02) -> None:
        self._send(_packet(id_, WRITE, bytes([addr]) + data))
        self._status(time.monotonic() + timeout_s)

    def sync_write(self, addr: int, rows: dict[int, bytes]) -> None:
        width = {len(v) for v in rows.values()}
        if len(width) != 1:
            raise BusError("sync write rows differ in width")
        (w,) = width
        params = bytes([addr, w]) + b"".join(bytes([i]) + d for i, d in rows.items())
        self._send(_packet(0xFE, SYNC_WRITE, params))

    def sync_read(self, addr: int, n: int, ids: list[int], timeout_s: float = 0.012) -> dict[int, bytes]:
        self._send(_packet(0xFE, SYNC_READ, bytes([addr, n]) + bytes(ids)))
        deadline = time.monotonic() + timeout_s
        out: dict[int, bytes] = {}
        while len(out) < len(ids):
            rid, err, data = self._status(deadline)
            if rid in ids and len(data) == n:
                out[rid] = data
        return out


class FeetechIO(RobotIO):
    def __init__(self, port: str, calibration: Calibration, echo: bool | str = "auto",
                 imu_source=None, control_dt: float = 0.02):
        calibration.validate()
        self.cal = calibration
        self.bus = FeetechBus(port, echo=echo)
        self.ids = [j.id for j in calibration.joints]
        self.imu_on_bus = imu_source is None
        self.read_ids = ([IMU_ID] if self.imu_on_bus else []) + self.ids  # IMU first, like official
        self.decoder = SflpDecoder(calibration.imu_mount)
        self.imu_source = imu_source
        self.control_dt = control_dt
        self._next_tick = time.monotonic()
        self._t0 = time.monotonic()
        self.feedback_raw: dict[int, bytes] = {}

    # -- conversions ------------------------------------------------------------------------
    def _decode(self, raw: dict[int, bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos, vel, volt, temp = (np.zeros(NUM_JOINTS) for _ in range(4))
        for k, j in enumerate(self.cal.joints):
            b = raw[j.id]
            p, s = struct.unpack_from("<HH", b, 0)
            pos[k] = j.to_rad(_signmag(p))
            vel[k] = _signmag(s) * SPEED_RAD_S_PER_COUNT * j.direction
            volt[k] = b[6] * 0.1
            temp[k] = b[7]
        return pos, vel, volt, temp

    def read(self) -> Sensors:
        raw = self.bus.sync_read(REG_FEEDBACK, FEEDBACK_LEN, self.read_ids)
        self.feedback_raw = raw
        pos, vel, self.voltages, self.temperatures = self._decode(raw)
        if self.imu_on_bus:
            imu = self.decoder.decode(raw[IMU_ID][:BLOCK_LEN])
            ready = self.decoder.ready
        else:
            imu, ready = self.imu_source.read()
        return Sensors(positions=pos, velocities=vel, imu=imu, imu_ready=ready, t=self.now())

    def now(self) -> float:
        return time.monotonic() - self._t0

    def write(self, targets: np.ndarray) -> None:
        rows = {}
        for k, j in enumerate(self.cal.joints):
            raw = j.to_raw(float(targets[k]))
            rows[j.id] = struct.pack("<H", _signmag_word(raw)) + b"\x00\x00\x00\x00"
        self.bus.sync_write(REG_GOAL, rows)

    def set_torque(self, on: bool) -> None:
        if on:
            # Volatile gains first, then goal = present position, then torque: no jump.
            self.bus.sync_write(REG_RAM_P, {i: bytes([FIRMWARE_P]) for i in self.ids})
            self.bus.sync_write(REG_RAM_D, {i: b"\x00" for i in self.ids})
            self.bus.sync_write(REG_TORQUE_LIMIT, {i: struct.pack("<H", TORQUE_LIMIT) for i in self.ids})
            p = self.bus.sync_read(REG_RAM_P, 1, self.ids)
            bad = [i for i in self.ids if p.get(i) != bytes([FIRMWARE_P])]
            if bad:
                raise BusError(f"P gain readback mismatch on {bad}; torque-on refused")
            s = self.read()
            self.write(s.positions)
        self.bus.sync_write(REG_TORQUE, {i: bytes([1 if on else 0]) for i in self.ids})

    def wait_tick(self) -> None:
        self._next_tick += self.control_dt
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:  # overrun: resynchronise instead of bursting
            self._next_tick = time.monotonic()

    def raw_imu_block(self) -> bytes | None:
        """The current 12-byte SFLP block, whichever way the IMU is wired (calibration tools)."""
        if self.imu_on_bus:
            return self.bus.sync_read(REG_FEEDBACK, FEEDBACK_LEN, [IMU_ID])[IMU_ID][:BLOCK_LEN]
        return self.imu_source.latest_block()

    def close(self) -> None:
        self.bus.close()
        if self.imu_source is not None:
            self.imu_source.close()
