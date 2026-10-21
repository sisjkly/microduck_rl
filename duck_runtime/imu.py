"""Body IMU: the 12-byte LSM6DSV16X SFLP block, decoded exactly like the official runtime.

Port of pollen-robotics/microduck duck-control/src/imu.rs (SflpDecoder). The block is what
the official ``imu_to_dxl`` v2 board and kissqy's FT6 firmware (same RevA PCB, Feetech
protocol) both serve:

  bytes 0..6   gyro x/y/z, i16 LE, +-500 dps (17.5 mdps/LSB)
  bytes 6..12  SFLP game-rotation quaternion x/y/z as IEEE half; w = sqrt(1 - |xyz|^2)

Any other IMU can feed the runtime through ``ImuData`` directly (see ``io.ImuSource``);
what matters is the output convention: gyro in the trunk frame (x fwd, y left, z up),
gravity = R_trunk->world^T [0, 0, -1] (upright reads [0, 0, -1]).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

GYRO_RAD_PER_LSB = 0.0175 * math.pi / 180.0  # imu.rs:48
BLOCK_LEN = 12

# Sensor->trunk mount, scalar-first [w, x, y, z] (imu.rs:78-88).
DEFAULT_MOUNT = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)  # official "zero3": +90 deg about Y
IDENTITY_MOUNT = (1.0, 0.0, 0.0, 0.0)                      # official "beta": board axes = trunk


@dataclass
class ImuData:
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotate(q, v) -> np.ndarray:
    """q * v * q^-1"""
    w, x, y, z = q
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return np.asarray(v) + w * t + np.cross(u, t)


def rotate_inverse(q, v) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return np.asarray(v) - w * t + np.cross(u, t)


def _normalise(v) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.asarray(v) / n if n > 0.1 else np.array([0.0, 0.0, -1.0])


def _half(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def encode_block(gyro_sensor_rad_s, quat_sensor) -> bytes:
    """Inverse of decode, for the simulated board (and tests)."""
    g = [int(max(-32768, min(32767, round(v / GYRO_RAD_PER_LSB)))) for v in gyro_sensor_rad_s]
    q = np.asarray(quat_sensor, dtype=float)
    if q[0] < 0:  # SFLP reconstructs w >= 0
        q = -q
    return struct.pack("<hhh", *g) + struct.pack("<eee", float(q[1]), float(q[2]), float(q[3]))


class SflpDecoder:
    """imu.rs SflpDecoder: mount rotation, last-good quaternion hold, 3-sample median."""

    READY_SAMPLES = 25  # ~0.25 s of fused output at 100 Hz

    def __init__(self, mount=DEFAULT_MOUNT):
        self.mount = np.asarray(mount, dtype=float)
        self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.quat_samples = 0
        self.gyro_history = [np.zeros(3), np.zeros(3)]
        self.gravity_history = [np.array([0.0, 0.0, -1.0])] * 2

    @property
    def ready(self) -> bool:
        return self.quat_samples >= self.READY_SAMPLES

    def decode(self, raw: bytes) -> ImuData:
        if len(raw) < BLOCK_LEN:
            raise ValueError(f"IMU block is {len(raw)} bytes, need {BLOCK_LEN}")
        gx, gy, gz = struct.unpack_from("<hhh", raw, 0)
        gyro = rotate(self.mount, np.array([gx, gy, gz], dtype=float) * GYRO_RAD_PER_LSB)

        packed = struct.unpack_from("<HHH", raw, 6)
        if packed != (0, 0, 0):  # all-zero = SFLP not running yet: keep last good
            x, y, z = (_half(p) for p in packed)
            n2 = x * x + y * y + z * z
            if all(math.isfinite(c) for c in (x, y, z)) and n2 <= 1.02:
                w = math.sqrt(max(0.0, 1.0 - n2))
                m = self.mount
                q = quat_mul([w, x, y, z], [m[0], -m[1], -m[2], -m[3]])
                n = float(np.linalg.norm(q))
                if n > 0.5:
                    self.last_quat = q / n
                    self.quat_samples += 1

        gravity = _normalise(rotate_inverse(self.last_quat, np.array([0.0, 0.0, -1.0])))
        out = ImuData(
            gyro=np.median(np.stack([*self.gyro_history, gyro]), axis=0),
            gravity=np.median(np.stack([*self.gravity_history, gravity]), axis=0),
            quat=self.last_quat.copy(),
        )
        self.gyro_history = [self.gyro_history[1], gyro]
        self.gravity_history = [self.gravity_history[1], gravity]
        return out
