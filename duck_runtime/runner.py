"""The control loop: limp -> ramp to HOME -> hold -> policy, with a fall exit.

States
  LIMP     torque off. Entered at start and after a fall (``on_fall="limp"``).
  RAMP     torque on, targets interpolated from the measured pose to DEFAULT_POSITION over
           ``ramp_s`` (official: interpolate_to, only on an explicit request).
  HOLD     torque on at DEFAULT_POSITION, policy not running. Waits for ``start()``.
  RUN      policy at 50 Hz.

Nothing moves on its own: every transition out of LIMP/HOLD is an explicit call, like
the official robotd ("the control loop must never move the robot on its own").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .contract import DEFAULT_POSITION, MOUTH_CLOSED, MOUTH_INDEX
from .controller import Command, Controller
from .io import RobotIO, Sensors
from .safety import Safety


class State(Enum):
    LIMP = "limp"
    RAMP = "ramp"
    HOLD = "hold"
    RUN = "run"


@dataclass
class TickLog:
    t: float
    state: str
    obs: np.ndarray | None
    targets: np.ndarray | None
    positions: np.ndarray
    gravity: np.ndarray
    gyro: np.ndarray
    fallen: bool


@dataclass
class Runner:
    io: RobotIO
    controller: Controller
    safety: Safety = field(default_factory=Safety)
    ramp_s: float = 2.0
    on_fall: str = "limp"          # "limp" | "keep" (keep driving, like the official default)
    log: list[TickLog] | None = None

    def __post_init__(self):
        self.state = State.LIMP
        self.command = Command()
        self._command_t = -1e9
        self._ramp_from: np.ndarray | None = None
        self._ramp_t0 = 0.0
        self.fall_events = 0
        self.refusal = ""

    # -- requests ------------------------------------------------------------------------
    def enable(self) -> None:
        """LIMP -> RAMP (torque on, ease to HOME)."""
        if self.state is not State.LIMP:
            return
        s = self.io.read()
        self.io.set_torque(True)
        self._ramp_from = s.positions.copy()
        self._ramp_t0 = s.t
        self.state = State.RAMP

    def start(self) -> bool:
        """HOLD -> RUN. Refused while the IMU is not ready or the robot reads as fallen."""
        if self.state is not State.HOLD:
            return False
        s = self.io.read()
        if not s.imu_ready or s.imu.gravity[2] > self.safety.fall_gravity_z:
            self.refusal = "IMU not ready" if not s.imu_ready else "robot is not upright"
            return False
        self.controller.reset()
        self.state = State.RUN
        return True

    def stop(self) -> None:
        """RUN -> HOLD (back to HOME, policy off)."""
        if self.state is State.RUN:
            s = self.io.read()
            self._ramp_from, self._ramp_t0 = s.positions.copy(), s.t
            self.state = State.RAMP

    def limp(self) -> None:
        self.io.set_torque(False)
        self.state = State.LIMP

    def set_command(self, command: Command, now: float) -> None:
        self.command = command
        self._command_t = now

    # -- one control period ----------------------------------------------------------------
    def tick(self) -> Sensors:
        s = self.io.read()
        fallen = self.safety.update_fall(s) if s.imu_ready else False
        obs = targets = None

        if self.state is State.RUN and fallen:
            self.fall_events += 1
            if self.on_fall == "limp":
                self.limp()

        if self.state is State.RAMP:
            a = min(1.0, (s.t - self._ramp_t0) / max(self.ramp_s, 1e-6))
            targets = (1 - a) * self._ramp_from + a * DEFAULT_POSITION
            targets[MOUTH_INDEX] = MOUTH_CLOSED
            if a >= 1.0:
                self.state = State.HOLD
        elif self.state is State.HOLD:
            targets = DEFAULT_POSITION.copy()
            targets[MOUTH_INDEX] = MOUTH_CLOSED
        elif self.state is State.RUN:
            if not s.imu_ready:
                raise RuntimeError("IMU not ready: refusing to run the policy on a default orientation")
            cmd = self.safety.gate_command(self.command, s.t - self._command_t)
            targets = self.controller.step(s, cmd)
            obs = self.controller.last_obs

        if targets is not None:
            self.io.write(self.safety.clamp(targets))
        if self.log is not None:
            self.log.append(TickLog(s.t, self.state.value, obs, None if targets is None else targets.copy(),
                                    s.positions.copy(), s.imu.gravity.copy(), s.imu.gyro.copy(), fallen))
        self.io.wait_tick()
        return s

    def run_for(self, seconds: float, realtime: bool = False) -> None:
        n = int(round(seconds / self.io.control_dt))
        for _ in range(n):
            t0 = time.monotonic()
            self.tick()
            if realtime:
                time.sleep(max(0.0, self.io.control_dt - (time.monotonic() - t0)))
