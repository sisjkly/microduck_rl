"""This robot's servo path: Pi --USB CDC--> ESP32-S3 "MDK2" bridge --UART--> 14x HD1910 (+ mouth).

The ESP32-S3 firmware (microduck_esp32_bridge, bridge_runtime.cpp) answers Dynamixel
Protocol 2.0 on its native USB port, one virtual XL330 per servo, and drives the Feetech
bus itself (polls feedback every 10 ms). Only this subset is used:

  PING                         virtual model 1200, bridge FW 2
  READ 256 x64   "MDK2" status: valid/config/torque masks, fault, reset reason, uptime...
  READ 320 x32   per-joint diagnostics: raw HD1910 position, speed, current, voltage,
                 temperature, error, firmware center/direction, mode, fresh/config flags
  READ 360 x16   "NPD1" native P/D state
  READ 1024+a    read-only window onto the servo's real register a (verifies RAM P/D)
  SYNC READ      of 320 x32 for every joint (one round trip per tick)
  SYNC WRITE 116 goal (virtual units, int32)   SYNC WRITE 64 torque
  INSTR 0xA1     "NPD1" p d: write native RAM P(50)/D(51) directly, torque must be off

Calibration lives here in Python, not in the firmware: the firmware maps virtual units
affinely, ``ft = center + dir * (dx - 2048)`` (bridge_protocol.h mapGoal/mapFeedback),
and we read ``center``/``dir`` back from READ 320, so the encoder count we want is sent
exactly whatever bridge_config.h says.

Firmware safety we rely on (bridge_runtime.cpp): torque drops and a fault latches on host
silence > 1 s, stale feedback > 100 ms, servo error, wrong mode or a goal outside its raw
limits. A torque-off written to every servo the bridge serves clears a latched fault.

The stock sketch serves IDs 1-14; esp32_bridge_patch/add_mouth.py adds the mouth (ID 15).
Which IDs answer is found by pinging at connect, so both firmwares work.

Pure stdlib + numpy (termios, no pyserial), so it runs in ~/microduck/.venv as is.
"""

from __future__ import annotations

import fcntl
import math
import os
import select
import struct
import termios
import time
import tty

import numpy as np

from .calibration import Calibration
from .contract import MOUTH_CLOSED, NUM_JOINTS
from .io import RobotIO, Sensors

BRIDGE_IDS = tuple(range(1, 16))   # IDs a bridge may serve: 1-14, plus 15 with the mouth firmware
ADDR_TORQUE, ADDR_GOAL = 64, 116
ADDR_STATUS, ADDR_DIAG, ADDR_NPD, NATIVE_BASE = 256, 320, 360, 1024
INS_PING, INS_READ, INS_SYNC_READ, INS_SYNC_WRITE, INS_NPD = 0x01, 0x02, 0x82, 0x83, 0xA1
HD_RAD_PER_COUNT = 2.0 * math.pi / 4096.0
SPEED_RAD_S_PER_COUNT = 50.0 * HD_RAD_PER_COUNT  # HD1910 speed unit: 50 steps/s
TRAINED_P, TRAINED_D = 5, 0  # what microduck_rl's HD1910 BAM fit was identified at

FAULTS = {0: "ok", 1: "host timeout", 2: "feedback missing/stale", 3: "servo error",
          4: "goal outside raw limits", 5: "servo mode is not 4", 6: "bus write failed",
          7: "invalid bridge_config.h", 8: "joint not marked calibrated", 9: "bus task failed",
          10: "calibration interlock", 11: "calibration tracking", 12: "calibration current"}


class BridgeError(RuntimeError):
    pass


# -- Dynamixel 2.0 framing (same CRC/stuffing as bridge_protocol.h) ----------------------
def _crc_table():
    out = []
    for v in range(256):
        c = v << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        out.append(c)
    return out


_CRC = _crc_table()


def crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c = ((c << 8) ^ _CRC[((c >> 8) ^ b) & 0xFF]) & 0xFFFF
    return c


def stuff(body: bytes) -> bytes:
    out = bytearray()
    for i, b in enumerate(body):
        out.append(b)
        if i >= 2 and body[i - 2:i + 1] == b"\xff\xff\xfd":
            out.append(0xFD)
    return bytes(out)


def unstuff(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        out.append(data[i])
        if out[-3:] == b"\xff\xff\xfd":
            i += 1
            if i >= len(data) or data[i] != 0xFD:
                raise BridgeError("missing stuffing byte")
        i += 1
    return bytes(out)


def packet(id_: int, ins: int, params: bytes = b"") -> bytes:
    body = stuff(bytes([ins]) + params)
    frame = b"\xff\xff\xfd\x00" + bytes([id_]) + struct.pack("<H", len(body) + 2) + body
    return frame + struct.pack("<H", crc16(frame))


class StatusParser:
    """Byte stream -> (id, error, params) status packets."""

    def __init__(self):
        self.buf = b""
        self.bad = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self.buf += data
        out = []
        while True:
            k = self.buf.find(b"\xff\xff\xfd\x00")
            if k < 0:
                self.buf = self.buf[-3:]
                return out
            self.buf = self.buf[k:]
            if len(self.buf) < 7:
                return out
            n = 7 + struct.unpack_from("<H", self.buf, 5)[0]
            if n < 11 or n > 600:
                self.buf, self.bad = self.buf[1:], self.bad + 1
                continue
            if len(self.buf) < n:
                return out
            frame, self.buf = self.buf[:n], self.buf[n:]
            if crc16(frame[:-2]) != struct.unpack_from("<H", frame, n - 2)[0]:
                self.bad += 1
                continue
            body = unstuff(frame[7:-2])
            if body[0] != 0x55:
                continue  # an instruction echo, never sent by the bridge; ignore
            out.append((frame[4], body[1], body[2:]))


class Port:
    """Raw, exclusive tty (USB CDC ignores the baud rate)."""

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            raise BridgeError(f"{path} is in use by another program "
                              "(stop it first: sudo systemctl stop microduck-gamepad.service)")
        try:
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
        except OSError:
            pass  # ptys in tests
        tty.setraw(self.fd, termios.TCSANOW)
        attr = termios.tcgetattr(self.fd)
        attr[4] = attr[5] = getattr(termios, "B1000000", termios.B115200)
        termios.tcsetattr(self.fd, termios.TCSANOW, attr)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            select.select([], [self.fd], [], 0.05)
            try:
                n = os.write(self.fd, view)
            except BlockingIOError:
                continue
            view = view[n:]

    def read_some(self, timeout: float) -> bytes:
        if not select.select([self.fd], [], [], max(0.0, timeout))[0]:
            return b""
        try:
            return os.read(self.fd, 4096)
        except BlockingIOError:
            return b""

    def drain(self) -> None:
        while self.read_some(0):
            pass

    def close(self) -> None:
        os.close(self.fd)


class Bridge:
    def __init__(self, path: str, timeout_s: float = 0.03):
        self.port = Port(path)
        self.parser = StatusParser()
        self.timeout_s = timeout_s

    def close(self):
        self.port.close()

    def _transact(self, pkt: bytes, expect: list[int], timeout_s: float | None = None) -> dict[int, tuple[int, bytes]]:
        self.port.drain()
        self.parser.buf = b""
        self.port.write(pkt)
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        got: dict[int, tuple[int, bytes]] = {}
        while len(got) < len(expect):
            left = deadline - time.monotonic()
            if left <= 0:
                missing = [i for i in expect if i not in got]
                raise BridgeError(f"no reply from virtual ID(s) {missing}")
            for id_, err, params in self.parser.feed(self.port.read_some(left)):
                if id_ in expect:
                    got[id_] = (err, params)
        return got

    def ping(self, id_: int) -> bool:
        try:
            self._transact(packet(id_, INS_PING), [id_])
            return True
        except BridgeError:
            return False

    def served_ids(self) -> list[int]:
        """The virtual IDs this firmware answers (PING): 1-14, or 1-15 with the mouth patch."""
        return [i for i in BRIDGE_IDS if self.ping(i)]

    def read(self, id_: int, addr: int, n: int, timeout_s: float | None = None) -> bytes:
        err, data = self._transact(packet(id_, INS_READ, struct.pack("<HH", addr, n)), [id_], timeout_s)[id_]
        if err or len(data) != n:
            raise BridgeError(f"read ID {id_} @{addr}x{n}: error 0x{err:02x}, {len(data)} B")
        return data

    def sync_read(self, addr: int, n: int, ids) -> dict[int, bytes]:
        ids = list(ids)
        got = self._transact(packet(0xFE, INS_SYNC_READ, struct.pack("<HH", addr, n) + bytes(ids)), ids)
        out = {}
        for i, (err, data) in got.items():
            if err or len(data) != n:
                raise BridgeError(f"sync read ID {i} @{addr}: error 0x{err:02x}")
            out[i] = data
        return out

    def sync_write(self, addr: int, width: int, rows: dict[int, bytes]) -> None:
        """Broadcast sync write: the bridge never answers it (Protocol 2.0)."""
        params = struct.pack("<HH", addr, width) + b"".join(bytes([i]) + d for i, d in rows.items())
        self.port.write(packet(0xFE, INS_SYNC_WRITE, params))

    def status(self, id_: int = BRIDGE_IDS[0]) -> dict:
        b = self.read(id_, ADDR_STATUS, 64)
        if b[:4] != b"MDK2":
            raise BridgeError(f"not an MDK2 bridge (got {b[:4]!r})")
        u16 = lambda o: struct.unpack_from("<H", b, o)[0]  # noqa: E731
        u32 = lambda o: struct.unpack_from("<I", b, o)[0]  # noqa: E731
        return dict(version=u16(4), valid_mask=u16(6), config_mask=u16(8), torque_mask=u16(10),
                    fault=u16(12), fault_name=FAULTS.get(u16(12), "unknown"), reset_reason=u16(14),
                    uptime_ms=u32(16), frames=u32(20), bad_frames=u32(24), echoes=u32(28),
                    incomplete_polls=u32(32), cycle_us=u32(36), host_age_ms=u32(44),
                    calibrated_mask=u16(48), stop_epoch=u32(50))

    def npd(self, id_: int, p: int, d: int) -> None:
        err, _ = self._transact(packet(id_, INS_NPD, b"NPD1" + bytes([p, d])), [id_])[id_]
        if err:
            raise BridgeError(f"NPD1 on ID {id_} refused (0x{err:02x}): torque must be off and no fault")

    def native(self, id_: int, addr: int, n: int) -> bytes:
        return self.read(id_, NATIVE_BASE + addr, n, timeout_s=0.25)

    def torque(self, ids, on: bool) -> None:
        self.sync_write(ADDR_TORQUE, 1, {i: bytes([1 if on else 0]) for i in ids})


def parse_diag(b: bytes) -> dict:
    raw, = struct.unpack_from("<i", b, 0)
    speed, current = struct.unpack_from("<hh", b, 8)
    center, = struct.unpack_from("<i", b, 20)
    flags = b[27]
    return dict(raw=raw, speed=speed, current_ma=current, voltage=b[12] * 0.1, temperature=b[14],
                error=b[15], center=center, direction=struct.unpack("b", b[24:25])[0], mode=b[25],
                fresh=bool(flags & 1), configured=bool(flags & 2), torque=bool(flags & 4),
                fw_calibrated=bool(flags & 8), ft_min=struct.unpack_from("<H", b, 28)[0],
                ft_max=struct.unpack_from("<H", b, 30)[0])


class BridgeIO(RobotIO):
    """RobotIO over the MDK2 bridge. A joint whose calibration id is 0 has no servo
    (the mouth before add_mouth.py): it reads as MOUTH_CLOSED and its targets are dropped."""

    def __init__(self, port: str, calibration: Calibration, imu_source, control_dt: float = 0.02,
                 gains: tuple[int, int] = (TRAINED_P, TRAINED_D)):
        calibration.validate()
        self.cal = calibration
        self.imu_source = imu_source
        self.control_dt = control_dt
        self.gains = gains
        self.k_of = [k for k, j in enumerate(calibration.joints) if j.id]
        self.ids = [calibration.joints[k].id for k in self.k_of]
        self.bridge = Bridge(port)
        self.served = self.bridge.served_ids()
        unknown = set(self.ids) - set(self.served)
        if unknown:
            self.bridge.close()
            hint = (" — the mouth needs the 15-servo bridge firmware (duck_runtime/esp32_bridge_patch/add_mouth.py)"
                    if unknown <= {15} else "")
            raise BridgeError(f"calibration uses IDs {sorted(unknown)} the bridge does not serve "
                              f"(it answers {self.served}){hint}")
        st = self.bridge.status()
        if st["version"] != 2:
            raise BridgeError(f"MDK2 version {st['version']}, this backend speaks version 2")
        if self.bridge.read(self.ids[0], ADDR_NPD, 16)[:4] != b"NPD1":
            raise BridgeError("bridge firmware lacks NPD1 (native P/D); flash the MicroDuck_XGO_r1 firmware")
        self.diag = {i: parse_diag(d) for i, d in self.bridge.sync_read(ADDR_DIAG, 32, self.ids).items()}
        self.fw_map = {i: (d["center"], d["direction"]) for i, d in self.diag.items()}
        self.original_pd: dict[int, bytes] | None = None
        self._next_tick = time.monotonic()
        self._t0 = time.monotonic()
        self.currents = np.zeros(NUM_JOINTS)
        self.voltages = np.zeros(NUM_JOINTS)
        self.temperatures = np.zeros(NUM_JOINTS)

    # -- raw encoder <-> virtual units ------------------------------------------------------
    def _dx(self, id_: int, ft: int) -> int:
        center, d = self.fw_map[id_]
        return 2048 + d * (ft - center)

    def raw_positions(self) -> dict[int, int]:
        """Raw HD1910 counts by servo ID (calibration tools)."""
        self.diag = {i: parse_diag(d) for i, d in self.bridge.sync_read(ADDR_DIAG, 32, self.ids).items()}
        return {i: d["raw"] for i, d in self.diag.items()}

    def read(self) -> Sensors:
        diag = {i: parse_diag(d) for i, d in self.bridge.sync_read(ADDR_DIAG, 32, self.ids).items()}
        self.diag = diag
        pos, vel = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
        for k, j in enumerate(self.cal.joints):
            if not j.id:
                pos[k] = MOUTH_CLOSED
                continue
            d = diag[j.id]
            if not d["fresh"]:
                raise BridgeError(f"{j.name} (ID {j.id}): bridge feedback stale — servo/bus problem")
            pos[k] = j.to_rad(d["raw"])
            vel[k] = d["speed"] * SPEED_RAD_S_PER_COUNT * j.direction
            self.currents[k], self.voltages[k], self.temperatures[k] = d["current_ma"], d["voltage"], d["temperature"]
        imu, ready = self.imu_source.read()
        return Sensors(positions=pos, velocities=vel, imu=imu, imu_ready=ready, t=self.now())

    def write(self, targets: np.ndarray) -> None:
        rows = {}
        for k in self.k_of:
            j = self.cal.joints[k]
            rows[j.id] = struct.pack("<i", self._dx(j.id, j.to_raw(float(targets[k]))))
        self.bridge.sync_write(ADDR_GOAL, 4, rows)

    def set_torque(self, on: bool) -> None:
        if not on:
            # Every served servo at once: also clears a latched bridge fault.
            self.bridge.torque(self.served, False)
            return
        st = self.bridge.status()
        if st["fault"]:
            raise BridgeError(f"bridge fault {st['fault']} ({st['fault_name']}); "
                              "torque off (key 0 / tools relax) clears it once the cause is fixed")
        if self.original_pd is None:
            self.original_pd = {i: self.bridge.native(i, 50, 2) for i in self.ids}
        p, d = self.gains
        for i in self.ids:
            self.bridge.npd(i, p, d)
        deadline = time.monotonic() + 0.5
        pending = set(self.ids)
        while pending and time.monotonic() < deadline:
            pending = {i for i in pending if self.bridge.native(i, 50, 2) != bytes([p, d])}
        if pending:
            raise BridgeError(f"RAM P/D readback != {p}/{d} on IDs {sorted(pending)}; torque-on refused")
        # Goal = where each joint is now, then torque: the bridge applies them in order.
        here = self.raw_positions()
        self.bridge.sync_write(ADDR_GOAL, 4, {i: struct.pack("<i", self._dx(i, here[i])) for i in self.ids})
        self.bridge.torque(self.ids, True)
        deadline = time.monotonic() + 0.3
        mask = sum(1 << (i - 1) for i in self.ids)
        while time.monotonic() < deadline:
            st = self.bridge.status()
            if st["fault"]:
                raise BridgeError(f"bridge refused torque: fault {st['fault']} ({st['fault_name']})")
            if st["torque_mask"] & mask == mask:
                return
            time.sleep(0.01)
        raise BridgeError("torque did not come on within 300 ms")

    def now(self) -> float:
        return time.monotonic() - self._t0

    def wait_tick(self) -> None:
        self._next_tick += self.control_dt
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = time.monotonic()

    def raw_imu_block(self) -> bytes | None:
        return self.imu_source.latest_block()

    def close(self) -> None:
        try:
            self.bridge.torque(self.served, False)
            if self.original_pd:  # leave the servos' RAM gains as we found them
                time.sleep(0.05)
                for i, pd in self.original_pd.items():
                    try:
                        self.bridge.npd(i, pd[0], pd[1])
                    except BridgeError:
                        pass
        finally:
            self.bridge.close()
            if self.imu_source is not None:
                self.imu_source.close()
