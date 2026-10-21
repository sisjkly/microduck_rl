"""The last gate before a motor: finite check, joint-limit clamp, fall report, deadman.

Same rules as the official duck-control/src/safety.rs, with one difference: the clamp is
per joint (calibrated mechanical limits) rather than the actuator's +-pi, because on a
clone the limits are measured, not assumed.

  - NaN / inf targets are refused (exception), never clamped.
  - Targets are clamped into [min, max] per joint.
  - Fall verdict: gravity_z > -0.5 (~60 deg) held 200 ms (safety.rs:72-73). A report;
    what to do about it is the runner's decision (default: limp).
  - Deadman: no fresh command for 0.5 s -> twist zeroed (safety.rs:74). Stop is not limp.
"""

from __future__ import annotations

import numpy as np

from .contract import JOINT_LIMITS
from .controller import Command
from .io import Sensors


class Safety:
    def __init__(self, limits: np.ndarray | None = None, fall_gravity_z: float = -0.5,
                 fall_debounce_s: float = 0.2, deadman_s: float = 0.5):
        self.limits = np.array(JOINT_LIMITS if limits is None else limits, dtype=float)
        self.fall_gravity_z = fall_gravity_z
        self.fall_debounce_s = fall_debounce_s
        self.deadman_s = deadman_s
        self._tilt_since: float | None = None
        self.fallen = False
        self.clamped_last = np.zeros(len(self.limits), dtype=bool)

    def clamp(self, targets: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(targets)):
            raise FloatingPointError(f"non-finite target refused: {targets}")
        out = np.clip(targets, self.limits[:, 0], self.limits[:, 1])
        self.clamped_last = out != targets
        return out

    def update_fall(self, sensors: Sensors) -> bool:
        if sensors.imu.gravity[2] > self.fall_gravity_z:
            if self._tilt_since is None:
                self._tilt_since = sensors.t
            self.fallen = sensors.t - self._tilt_since >= self.fall_debounce_s
        else:
            self._tilt_since = None
            self.fallen = False
        return self.fallen

    def gate_command(self, command: Command, command_age_s: float) -> Command:
        if command_age_s > self.deadman_s:
            return Command(twist=(0.0, 0.0, 0.0), head=command.head)
        return command
