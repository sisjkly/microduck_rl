"""Per-robot calibration: joint zero/direction/limits and the IMU mount. JSON on disk.

    rad = (raw - zero_raw) * 2pi/4096 * direction

``zero_raw`` is the encoder count at the joint's MJCF zero, ``direction`` makes positive
radians turn the way the MJCF joint axis does. Both depend on how each horn was fitted,
so they are measured on *this* robot (``python -m duck_runtime.tools zero`` and the
direction check), never copied. The template's directions are kissqy's measured ones
for an official skeleton with HD1910s — a starting guess to verify joint by joint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .contract import JOINT_IDS, JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS
from .imu import DEFAULT_MOUNT

COUNTS_PER_RAD = 4096.0 / (2.0 * math.pi)

# kissqy hardware.json (official skeleton + HD1910, 2026-10-03), JOINT_NAMES order.
TEMPLATE_DIRECTIONS = (1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1)


@dataclass
class JointCal:
    name: str
    id: int
    zero_raw: int = 2048
    direction: int = 1
    min_rad: float = -math.pi
    max_rad: float = math.pi
    calibrated: bool = False

    def to_rad(self, raw: int) -> float:
        return (raw - self.zero_raw) / COUNTS_PER_RAD * self.direction

    def to_raw(self, rad: float) -> int:
        if not math.isfinite(rad):
            raise ValueError(f"{self.name}: non-finite target")
        rad = min(max(rad, self.min_rad), self.max_rad)
        raw = round(self.zero_raw + rad * self.direction * COUNTS_PER_RAD)
        if not 0 <= raw <= 4095:
            raise ValueError(f"{self.name}: target {rad:.3f} rad -> raw {raw} outside single turn")
        return raw


@dataclass
class Calibration:
    joints: list[JointCal]
    imu_mount: list[float] = field(default_factory=lambda: list(DEFAULT_MOUNT))
    imu_mount_verified: bool = False
    notes: str = ""

    @classmethod
    def template(cls) -> "Calibration":
        return cls(joints=[
            JointCal(name=n, id=i, direction=d, min_rad=float(lo), max_rad=float(hi))
            for n, i, d, (lo, hi) in zip(JOINT_NAMES, JOINT_IDS, TEMPLATE_DIRECTIONS, JOINT_LIMITS)
        ], notes="TEMPLATE: zero_raw/direction unmeasured, imu_mount unverified")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(joints=[JointCal(**j) for j in d["joints"]], imu_mount=d["imu_mount"],
                   imu_mount_verified=d.get("imu_mount_verified", False), notes=d.get("notes", ""))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    def limits(self) -> np.ndarray:
        return np.array([(j.min_rad, j.max_rad) for j in self.joints])

    def validate(self, require_calibrated: bool = False) -> None:
        if [j.name for j in self.joints] != list(JOINT_NAMES):
            raise ValueError("calibration joints must be the 15 JOINT_NAMES in order")
        if len({j.id for j in self.joints}) != NUM_JOINTS:
            raise ValueError("duplicate servo IDs in calibration")
        for j, (lo, hi) in zip(self.joints, JOINT_LIMITS):
            if j.direction not in (-1, 1) or not 0 <= j.zero_raw <= 4095:
                raise ValueError(f"{j.name}: bad direction/zero_raw")
            if not (lo - 1e-6 <= j.min_rad < j.max_rad <= hi + 1e-6):
                raise ValueError(f"{j.name}: limits must lie inside the MJCF range [{lo}, {hi}]")
            if require_calibrated and not j.calibrated:
                raise ValueError(f"{j.name}: not calibrated")
        n = math.sqrt(sum(v * v for v in self.imu_mount))
        if abs(n - 1.0) > 1e-6:
            raise ValueError("imu_mount must be a unit quaternion [w, x, y, z]")
        if require_calibrated and not self.imu_mount_verified:
            raise ValueError("imu_mount not verified (python -m duck_runtime.tools imu-mount)")
