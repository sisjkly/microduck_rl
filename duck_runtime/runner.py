"""The control loop: limp -> ramp to HOME -> hold -> policy, with a fall exit.

States
  LIMP     torque off. Entered at start and after a fall (``on_fall="limp"``).
  RAMP     torque on, targets interpolated from the measured pose to DEFAULT_POSITION over
           ``ramp_s`` (official: interpolate_to, only on an explicit request).
  HOLD     torque on at DEFAULT_POSITION, policy not running. Waits for ``start()``.
  RUN      policy at 50 Hz.

Nothing moves on its own: every transition out of LIMP/HOLD is an explicit call, like
the official robotd ("the control loop must never move the robot on its own").

Inside RUN, with a policy set (``policy_set.py``), the network is picked per tick exactly as
robotd/src/control.rs:540-640 does: one-shot skill > sit/rise > walk. One move at a time and
none from the seat; while a move runs or the robot sits the fall verdict does not limp it
(control.rs ``busy()`` gates the limp-fall reflex: a roulade is upside down by design).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
import math

import numpy as np

from .contract import DEFAULT_POSITION, HEAD_JOINTS, MOUTH_CLOSED, MOUTH_INDEX
from .controller import Command, Controller
from .io import RobotIO, Sensors
from .policy_set import PolicySet, Skill
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
    currents: np.ndarray | None = None   # mA per joint, when the backend reports it (bridge)
    mode: str = ""                       # network driving in RUN: walk / sit / rise / <skill>


@dataclass
class _Move:
    skill: Skill
    remaining: float
    elapsed: float = 0.0
    unwinding: bool = False


@dataclass
class Runner:
    io: RobotIO
    controller: Controller
    safety: Safety = field(default_factory=Safety)
    ramp_s: float = 2.0
    on_fall: str = "limp"          # "limp" | "keep" (keep driving, like the official default)
    log: list[TickLog] | None = None
    # Neck/head offsets from HOME (rad, [neck_pitch, head_pitch, head_yaw, head_roll]) used for
    # RAMP/HOLD. Pair it with the same Command.head so the policy tracks the pose it starts in
    # (a build whose neck cannot reach HOME's +20 deg back).
    head_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    policies: PolicySet | None = None
    # Hand the rise to the gait only once the trunk has settled: within this tilt of upright
    # and below this pitch/roll rate, checked after rise_s, at most rise_extra_max_s longer.
    # Not in robotd (fixed rise_s): on this heavier-headed build one rise in four was still
    # swaying back 9 deg at 1.0 s, and the gait caught it with a lunge forward (set1.npz).
    rise_settle_tilt_deg: float | None = 6.0
    rise_settle_rate_dps: float = 30.0
    rise_extra_max_s: float = 2.0

    def __post_init__(self):
        self.state = State.LIMP
        self.command = Command()
        self._command_t = -1e9
        self._ramp_from: np.ndarray | None = None
        self._ramp_t0 = 0.0
        self.fall_events = 0
        self.refusal = ""
        self.stand_pose = DEFAULT_POSITION.copy()
        self.stand_pose[list(HEAD_JOINTS)] += np.asarray(self.head_offset, dtype=float)
        self._clear_moves()

    def _clear_moves(self) -> None:
        self.move: _Move | None = None
        self.sit = "up"              # "up" | "sitting" | "rising"
        self.sit_settle = 0.0
        self.rise_left = 0.0
        self.mode = ""

    @property
    def busy(self) -> bool:
        """control.rs busy(): a move in flight (the seat itself is parked, not busy)."""
        return self.move is not None or self.sit == "rising"

    def _move_blocked(self) -> str | None:
        if self.state is not State.RUN:
            return "policy not running (press 2 first)"
        if self.move is not None:
            return f"{self.move.skill.name} is still running"
        if self.sit == "sitting":
            return "the robot is sitting: stand it up first"
        if self.sit == "rising":
            return "the robot is standing up"
        return None

    def trigger(self, name: str) -> str | None:
        """Start a one-shot skill. None = started, else the refusal."""
        if self.policies is None or name not in self.policies.skills:
            return f"no skill {name!r} loaded"
        reason = self._move_blocked()
        if reason:
            return reason
        sk = self.policies.skills[name]
        self.move = _Move(sk, remaining=sk.duration_s)
        return None

    def sit_toggle(self) -> str | None:
        """Sit if standing, stand if seated (control.rs sit_toggle). None = done, else refusal."""
        if self.state is not State.RUN:
            return "policy not running (press 2 first)"
        if self.move is not None:
            return f"{self.move.skill.name} is still running"
        if self.sit == "up":
            if self.policies is None or self.policies.sitstand is None:
                return "no sit/stand policy loaded"
            self.sit, self.sit_settle = "sitting", self.policies.sitstand.ramp_s
            return None
        if self.sit == "sitting":
            if self.sit_settle > 0:
                return "still sitting down"
            self.sit, self.rise_left = "rising", self.policies.sitstand.rise_s
            return None
        return "already standing up"

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
        self._clear_moves()
        self.state = State.RUN
        return True

    def stop(self) -> None:
        """RUN -> HOLD (back to HOME, policy off)."""
        if self.state is State.RUN:
            s = self.io.read()
            self._ramp_from, self._ramp_t0 = s.positions.copy(), s.t
            self.state = State.RAMP
            self._clear_moves()

    def limp(self) -> None:
        self.io.set_torque(False)
        self.state = State.LIMP
        self._clear_moves()

    def set_command(self, command: Command, now: float) -> None:
        self.command = command
        self._command_t = now

    # -- one control period ----------------------------------------------------------------
    def tick(self) -> Sensors:
        s = self.io.read()
        fallen = self.safety.update_fall(s) if s.imu_ready else False
        obs = targets = None

        if self.state is State.RUN and fallen and not (self.busy or self.sit == "sitting"):
            self.fall_events += 1
            if self.on_fall == "limp":
                self.limp()

        if self.state is State.RAMP:
            a = min(1.0, (s.t - self._ramp_t0) / max(self.ramp_s, 1e-6))
            targets = (1 - a) * self._ramp_from + a * self.stand_pose
            targets[MOUTH_INDEX] = MOUTH_CLOSED
            if a >= 1.0:
                self.state = State.HOLD
        elif self.state is State.HOLD:
            targets = self.stand_pose.copy()
            targets[MOUTH_INDEX] = MOUTH_CLOSED
        elif self.state is State.RUN and not s.imu_ready:
            # Never run the policy on a stale/default orientation: go limp, say why.
            self.refusal = "IMU stopped updating while running"
            self.limp()
        elif self.state is State.RUN:
            targets = self._policy_step(s)
            obs = self.controller.last_obs

        if targets is not None:
            self.io.write(self.safety.clamp(targets))
        if self.log is not None:
            cur = getattr(self.io, "currents", None)
            self.log.append(TickLog(s.t, self.state.value, obs, None if targets is None else targets.copy(),
                                    s.positions.copy(), s.imu.gravity.copy(), s.imu.gyro.copy(), fallen,
                                    None if cur is None else np.array(cur, dtype=float),
                                    self.mode if self.state is State.RUN else ""))
        self.io.wait_tick()
        return s

    def _rise_settled(self, s: Sensors) -> bool:
        if self.rise_settle_tilt_deg is None or self.rise_left <= -self.rise_extra_max_s:
            return True
        tilt = math.degrees(math.acos(float(np.clip(-s.imu.gravity[2], -1.0, 1.0))))
        rate = math.degrees(float(np.hypot(s.imu.gyro[0], s.imu.gyro[1])))
        return tilt <= self.rise_settle_tilt_deg and rate <= self.rise_settle_rate_dps

    def _policy_step(self, s: Sensors) -> np.ndarray:
        dt = self.io.control_dt
        # Expire windows first, so a tick after the deadline runs the next thing (control.rs).
        if self.move is not None and self.move.remaining <= 0:
            sk = self.move.skill
            if not self.move.unwinding and sk.unwind_s > 0:
                self.move = replace(self.move, unwinding=True, remaining=sk.unwind_s)
            else:
                self.move = None
        if self.sit == "rising" and self.rise_left <= 0 and self._rise_settled(s):
            self.sit = "up"

        cmd = self.safety.gate_command(self.command, s.t - self._command_t)
        policy = scale = None
        if self.move is not None:
            # Head and body zeroed: every one-shot declares them unused (zero_command_padding).
            sk = self.move.skill
            if sk.encoding == "phase":
                ang = 2 * math.pi * self.move.elapsed / sk.period_s
                twist = (math.cos(ang), math.sin(ang), 0.0)
            else:
                twist = sk.unwind if self.move.unwinding else sk.twist
            cmd, policy, scale = Command(twist=twist), sk.policy, sk.action_scale
            self.mode = sk.name + (":unwind" if self.move.unwinding else "")
        elif self.sit in ("sitting", "rising"):
            ss = self.policies.sitstand
            # The posture flag rides twist vx: 1 sit, 0 stand; head stays live.
            cmd = replace(cmd, twist=(1.0, 0.0, 0.0) if self.sit == "sitting" else (0.0, 0.0, 0.0))
            policy, scale, self.mode = ss.policy, ss.action_scale, "sit" if self.sit == "sitting" else "rise"
        else:
            self.mode = "walk"
        targets = self.controller.step(s, cmd, policy, scale)

        # Advance the windows after the tick that used them.
        if self.move is not None:
            self.move.remaining -= dt
            self.move.elapsed += dt
        if self.sit == "rising":
            self.rise_left -= dt
        self.sit_settle = max(0.0, self.sit_settle - dt)
        return targets

    def run_for(self, seconds: float, realtime: bool = False) -> None:
        n = int(round(seconds / self.io.control_dt))
        for _ in range(n):
            t0 = time.monotonic()
            self.tick()
            if realtime:
                time.sleep(max(0.0, self.io.control_dt - (time.monotonic() - t0)))
