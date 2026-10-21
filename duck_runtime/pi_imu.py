"""LSM6DSV16X on the Pi's I2C bus -> the official 12-byte SFLP block -> SflpDecoder.

Same chip and the same configuration as the official imu_to_dxl board / FT6 firmware
(ST driver calls mirrored register by register, read-modify-write where the driver does):

  XL +-4 g, GY +-500 dps, XL/GY ODR 120 Hz high-performance, BDU + auto-increment,
  SFLP game rotation at 120 Hz batched into the FIFO (stream mode), XL/GY not batched,
  gyro bias zero (power-on default after SW_POR).

Each ``read()`` (one per control tick) drains the FIFO, keeps the newest game-rotation
quaternion (x/y/z half floats, exactly bytes 6..12 of the official block), reads the gyro
output registers (bytes 0..6) and decodes with the official mount/median logic.

Pure stdlib (ioctl I2C_RDWR), no smbus2 needed.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import time

from .imu import DEFAULT_MOUNT, ImuData, SflpDecoder

ADDRS = (0x6A, 0x6B)
WHO_AM_I, CHIP_ID = 0x0F, 0x70
FUNC_CFG_ACCESS = 0x01
FIFO_CTRL3, FIFO_CTRL4 = 0x09, 0x0A
CTRL1, CTRL2, CTRL3, CTRL6, CTRL8 = 0x10, 0x11, 0x12, 0x15, 0x17
FIFO_STATUS1, OUTX_L_G, FIFO_DATA_OUT_TAG = 0x1B, 0x22, 0x78
EMB_FUNC_EN_A, EMB_FUNC_FIFO_EN_A, SFLP_ODR = 0x04, 0x44, 0x5E
TAG_GAME_RV = 0x13
SW_POR, EMB_ACCESS = 0x04, 0x80
ODR_120HZ, SFLP_120HZ = 0x6, 0x3
FIFO_STREAM = 0x6
MAX_DRAIN = 16          # ~45 ms of SFLP at 120 Hz; a longer backlog is reset, not replayed
QUAT_MAX_AGE_S = 0.10   # like the official 100 ms IMU freshness window
ERRORS_BEFORE_REINIT = 10

I2C_SLAVE, I2C_RDWR, I2C_M_RD = 0x0703, 0x0707, 0x0001


class _Msg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("len", ctypes.c_uint16), ("buf", ctypes.c_void_p)]


class _Rdwr(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(_Msg)), ("nmsgs", ctypes.c_uint32)]


class LinuxI2C:
    """Register access on /dev/i2c-N (repeated-start reads)."""

    def __init__(self, bus: int, addr: int):
        self.addr = addr
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR | os.O_CLOEXEC)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def read(self, reg: int, n: int) -> bytes:
        a = (ctypes.c_uint8 * 1)(reg)
        out = (ctypes.c_uint8 * n)()
        msgs = (_Msg * 2)(_Msg(self.addr, 0, 1, ctypes.addressof(a)),
                          _Msg(self.addr, I2C_M_RD, n, ctypes.addressof(out)))
        fcntl.ioctl(self.fd, I2C_RDWR, bytes(_Rdwr(msgs, 2)))  # buffers stay alive: locals
        return bytes(out)

    def write(self, reg: int, value: int) -> None:
        if os.write(self.fd, bytes([reg, value & 0xFF])) != 2:
            raise OSError("short I2C write")

    def close(self) -> None:
        os.close(self.fd)


def open_lsm6dsv(bus: int = 1):
    errors = []
    for addr in ADDRS:
        try:
            dev = LinuxI2C(bus, addr)
        except OSError as e:
            errors.append(f"0x{addr:02X}: {e}")
            continue
        try:
            if dev.read(WHO_AM_I, 1)[0] == CHIP_ID:
                return dev
            errors.append(f"0x{addr:02X}: WHO_AM_I mismatch")
        except OSError as e:
            errors.append(f"0x{addr:02X}: {e}")
        dev.close()
    raise RuntimeError(f"no LSM6DSV16X on /dev/i2c-{bus}: " + "; ".join(errors))


class PiImu:
    """IMU source for BridgeIO: ``read() -> (ImuData, ready)``, ``latest_block()``, ``close()``."""

    def __init__(self, dev=None, bus: int = 1, mount=DEFAULT_MOUNT, clock=time.monotonic):
        self.dev = dev if dev is not None else open_lsm6dsv(bus)
        self.decoder = SflpDecoder(mount)
        self.clock = clock
        self.quat_raw = b"\x00" * 6
        self.quat_t = -1e9
        self.block: bytes | None = None
        self.errors = 0
        self.quat_count = 0
        self.resets = 0
        self.configure()

    # -- chip setup ---------------------------------------------------------------------------
    def _rmw(self, reg: int, mask: int, value: int) -> None:
        self.dev.write(reg, (self.dev.read(reg, 1)[0] & ~mask) | (value & mask))

    def configure(self) -> None:
        d = self.dev
        d.write(FUNC_CFG_ACCESS, SW_POR)              # full reboot incl. embedded functions
        time.sleep(0.03)
        for _ in range(30):
            try:
                if not d.read(FUNC_CFG_ACCESS, 1)[0] & SW_POR:
                    break
            except OSError:
                pass
            time.sleep(0.005)
        else:
            raise RuntimeError("LSM6DSV16X SW_POR did not complete")
        if d.read(WHO_AM_I, 1)[0] != CHIP_ID:
            raise RuntimeError("LSM6DSV16X WHO_AM_I mismatch after reset")
        self._rmw(CTRL3, 0x44, 0x44)                   # BDU | IF_INC
        self._rmw(CTRL8, 0x03, 0x01)                   # FS_XL = 4 g
        self._rmw(CTRL6, 0x0F, 0x02)                   # FS_G = 500 dps
        d.write(FIFO_CTRL3, 0x00)                      # XL/GY not batched: FIFO holds SFLP only
        self._rmw(FIFO_CTRL4, 0x07, FIFO_STREAM)
        self._rmw(CTRL1, 0x7F, ODR_120HZ)              # high-performance, 120 Hz
        self._rmw(CTRL2, 0x7F, ODR_120HZ)
        d.write(FUNC_CFG_ACCESS, EMB_ACCESS)
        try:
            self._rmw(SFLP_ODR, 0x38, SFLP_120HZ << 3)
            self._rmw(EMB_FUNC_FIFO_EN_A, 0x32, 0x02)  # game rotation only (no gravity/gbias)
            self._rmw(EMB_FUNC_EN_A, 0x02, 0x02)       # SFLP game rotation on
            if (d.read(SFLP_ODR, 1)[0] >> 3) & 7 != SFLP_120HZ or not d.read(EMB_FUNC_EN_A, 1)[0] & 2:
                raise RuntimeError("LSM6DSV16X embedded SFLP registers did not read back")
        finally:
            d.write(FUNC_CFG_ACCESS, 0x00)
        if d.read(CTRL1, 1)[0] & 0x0F != ODR_120HZ or d.read(CTRL6, 1)[0] & 0x0F != 0x02:
            raise RuntimeError("LSM6DSV16X main registers did not read back")

    def _reset_fifo(self) -> None:
        self._rmw(FIFO_CTRL4, 0x07, 0)                 # bypass empties it
        self._rmw(FIFO_CTRL4, 0x07, FIFO_STREAM)
        self.resets += 1

    # -- per tick ------------------------------------------------------------------------------
    def _poll(self) -> None:
        st = self.dev.read(FIFO_STATUS1, 2)
        level = st[0] | ((st[1] & 1) << 8)
        if level > MAX_DRAIN:
            self._reset_fifo()
        else:
            for _ in range(level):
                r = self.dev.read(FIFO_DATA_OUT_TAG, 7)
                if (r[0] >> 3) == TAG_GAME_RV:
                    self.quat_raw, self.quat_t = r[1:7], self.clock()
                    self.quat_count += 1
        gyro = self.dev.read(OUTX_L_G, 6)
        self.block = gyro + self.quat_raw

    def read(self) -> tuple[ImuData, bool]:
        try:
            self._poll()
            self.errors = 0
        except OSError:
            self.errors += 1
            if self.errors >= ERRORS_BEFORE_REINIT:
                try:
                    self.configure()  # restarts the fusion: readiness is re-earned below
                    self.decoder = SflpDecoder(self.decoder.mount)
                    self.quat_raw, self.quat_t, self.errors = b"\x00" * 6, -1e9, 0
                except (OSError, RuntimeError):
                    pass
            return self.decoder.decode(self.block or b"\x00" * 12), False
        imu = self.decoder.decode(self.block)
        fresh = self.clock() - self.quat_t <= QUAT_MAX_AGE_S
        return imu, bool(self.decoder.ready and fresh)

    def latest_block(self) -> bytes | None:
        try:
            self._poll()
        except OSError:
            return None
        return self.block if self.quat_count else None

    def close(self) -> None:
        self.dev.close()
