"""What the controller reads and writes, independent of the body behind it.

Implementations: ``bridge.BridgeIO`` (ESP32-S3 MDK2 bridge + Pi I2C IMU, this robot),
``sim_io.SimIO`` (MuJoCo + BAM HD1910, CPU) and
``feetech.FeetechIO`` (HD1910 servos + IMU on a Feetech TTL bus). The controller,
safety layer and policy code are shared, so a run in simulation exercises the exact
code path the robot runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .contract import NUM_JOINTS
from .imu import ImuData


@dataclass
class Sensors:
    """One tick of body state, in joint space (15 joints, JOINT_NAMES order, radians)."""
    positions: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    imu: ImuData = field(default_factory=ImuData)
    imu_ready: bool = False
    t: float = 0.0  # seconds, monotonic


class RobotIO(ABC):
    """The only object allowed to move a motor."""

    control_dt: float = 0.02

    @abstractmethod
    def read(self) -> Sensors: ...

    @abstractmethod
    def write(self, targets: np.ndarray) -> None:
        """Send 15 joint position targets (radians). Must be pre-clamped by Safety."""

    @abstractmethod
    def set_torque(self, on: bool) -> None: ...

    @abstractmethod
    def now(self) -> float:
        """Seconds on the same clock as Sensors.t (used for command age / deadman)."""

    def wait_tick(self) -> None:
        """Block until the next control period (the sim advances physics instead)."""

    def close(self) -> None:
        pass
