"""Host side of the ESP32 IMU bridge (duck_runtime/esp32_imu): frames -> ImuData.

A reader thread keeps the newest valid frame; the control loop calls ``read()`` once per
tick and decodes that frame's 12-byte block with the same ``SflpDecoder`` the official
runtime uses, so the median filter and quaternion hold behave per control tick exactly
as on the official robot.

Ready means: the decoder has seen 25 live quaternions, the newest frame is < 50 ms old,
a fresh quaternion arrived within the last 100 ms (SFLP runs at 120 Hz), and the ESP32
reported no sensor error. The runner refuses to start, and stops a running policy
(limp), while the IMU is not ready.

Frame (17 B): A5 5A | seq | flags | block[12] | crc8(seq..block, poly 0x07)
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .imu import BLOCK_LEN, DEFAULT_MOUNT, ImuData, SflpDecoder

MAGIC = b"\xa5\x5a"
FRAME_LEN = 17
FLAG_FRESH, FLAG_SFLP_SEEN, FLAG_ERROR = 1, 2, 4
MAX_FRAME_AGE_S = 0.05
MAX_QUAT_AGE_S = 0.1


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def make_frame(seq: int, flags: int, block: bytes) -> bytes:
    body = bytes([seq & 0xFF, flags & 0xFF]) + block
    return MAGIC + body + bytes([crc8(body)])


class FrameParser:
    """Byte stream -> (seq, flags, block). Resynchronises on the magic after noise."""

    def __init__(self):
        self.buf = b""
        self.bad_crc = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                self.buf = self.buf[-1:]
                return out
            if len(self.buf) - i < FRAME_LEN:
                self.buf = self.buf[i:]
                return out
            f = self.buf[i:i + FRAME_LEN]
            if crc8(f[2:16]) == f[16]:
                out.append((f[2], f[3], bytes(f[4:16])))
                self.buf = self.buf[i + FRAME_LEN:]
            else:
                self.bad_crc += 1
                self.buf = self.buf[i + 1:]


class Esp32Imu:
    def __init__(self, port: str = "/dev/serial0", baud: int = 921600, mount=DEFAULT_MOUNT,
                 clock=time.monotonic, serial_obj=None):
        if serial_obj is None:
            import serial
            serial_obj = serial.Serial(port, baud, timeout=0.01)
        self.ser = serial_obj
        self.clock = clock
        self.decoder = SflpDecoder(mount)
        self.parser = FrameParser()
        self._lock = threading.Lock()
        self._latest: tuple[int, int, bytes] | None = None
        self._t_frame = -1e9
        self._t_quat = -1e9
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="esp32-imu", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            data = self.ser.read(max(1, getattr(self.ser, "in_waiting", 0) or 1))
            if data:
                self.ingest(data)

    def ingest(self, data: bytes) -> None:
        for seq, flags, block in self.parser.feed(data):
            now = self.clock()
            with self._lock:
                self._latest = (seq, flags, block)
                self._t_frame = now
                if flags & FLAG_FRESH:
                    self._t_quat = now
                self.frames += 1

    def latest_block(self) -> bytes | None:
        with self._lock:
            return None if self._latest is None else self._latest[2]

    def read(self) -> tuple[ImuData, bool]:
        now = self.clock()
        with self._lock:
            latest, t_frame, t_quat = self._latest, self._t_frame, self._t_quat
        if latest is None:
            return ImuData(), False
        seq, flags, block = latest
        imu = self.decoder.decode(block[:BLOCK_LEN])
        ready = (self.decoder.ready and now - t_frame < MAX_FRAME_AGE_S and now - t_quat < MAX_QUAT_AGE_S
                 and bool(flags & FLAG_SFLP_SEEN) and not flags & FLAG_ERROR)
        return imu, ready

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.5)
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass
