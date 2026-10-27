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
from .safety import FallPredictor, Safety


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
    voltages: np.ndarray | None = None   # V per servo (bridge): sag under load slows big moves
    on_fall: str = ""                    # limp / keep / getup at this tick (the dashboard can switch it)


# Ground pick beak, scripted on the training phases (microduck_ground_pick_env_cfg.py:
# descent 0-0.375, hold at the bottom 0.375-0.425, rise with a payload at the beak after).
# The policy has no mouth output and robotd never opens it during a pick; this is ours.
PICK_MOUTH_OPEN = math.radians(25.0)


def pick_mouth(phase: float) -> float:
    """Beak target (rad) at ground-pick phase: open on the way down, close at the bottom."""
    closed, opened = MOUTH_CLOSED, PICK_MOUTH_OPEN
    if phase < 0.10:
        return closed
    if phase < 0.30:                      # open while descending
        return closed + (opened - closed) * (phase - 0.10) / 0.20
    if phase < 0.375:
        return opened
    if phase < 0.425:                     # bite during the hold at the bottom
        return opened + (closed - opened) * (phase - 0.375) / 0.05
    return closed


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
    # What a fall does while the policy runs:
    #   "limp"  torque off and stay off (this runtime's default)
    #   "keep"  keep driving; velstand v8 gets itself up (robotd with limp_fall = false)
    #   "getup" robotd's default limp-fall: torque off as the fall starts, land soft, wait
    #           until still, ease into the stand pose, hand back to the policy to stand up
    on_fall: str = "limp"
    log: list[TickLog] | None = None
    # Neck/head offsets from HOME (rad, [neck_pitch, head_pitch, head_yaw, head_roll]) used for
    # RAMP/HOLD. Pair it with the same Command.head so the policy tracks the pose it starts in
    # (a build whose neck cannot reach HOME's +20 deg back).
    head_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    policies: PolicySet | None = None
    # After a move ends the fall verdict waits this long before limping: a roulade hands back
    # at 1.0 s still ~73 deg over and velstand (v8) finishes getting up by ~2.5 s (SimIO,
    # 835 g). robotd keeps driving after a fall by default, so it never needed this.
    post_move_grace_s: float = 2.5
    # on_fall="keep": velstand v8 gets itself up (SimIO 835 g: 0.6-1.8 s from face/side/back).
    # Still down after this long -> limp, so a stuck robot does not thrash on the floor.
    getup_timeout_s: float | None = 5.0
    # robotd-params [safety] limp_fall_* defaults (landing and the pose back to standing).
    limp_fall_still_rate: float = 1.0     # rad/s: below this counts as landed
    limp_fall_still_s: float = 0.2
    limp_fall_max_s: float = 1.5          # stop waiting for stillness after this
    limp_fall_pose_s: float = 0.6
    # on_fall="getup", beyond robotd: a robot lying still is re-posed (ease into the stand pose,
    # hand back to the gait). velstand has fixed points on the floor: dash_20261027-071329.npz
    # lay on its side 5 s with the gait outputting one constant pose; SimIO started from that
    # exact state reproduces it (any target clamp) and one re-pose stands it up (0.5-2.9 s,
    # both sets, ideal and full-pack supply). Covers what limp-fall cannot: starting the policy
    # on the floor, and a fall inside a move (no limp-fall during moves, as robotd).
    # dash_20261027-083331.npz (new head): a re-pose with torque on left it propped on its
    # right side by its own stiff legs, 3 times; the one get-up that day went through the
    # limp phase (legs folded, it rolled onto its front). So a re-pose is the whole robotd
    # limp-fall sequence: torque off, at least repose_limp_min_s for the legs to fold and the
    # body to settle, torque on, pose, gait. Stuck = lying still, or no progress at all.
    stuck_rate: float = 0.3               # rad/s: |gyro| below this counts as lying still
    stuck_repose_s: float = 1.0           # still this long while down -> re-pose
    stuck_noprogress_s: float = 2.5       # or down this long since the gait took over
    repose_limp_min_s: float = 0.5
    max_reposes: int = 3                  # then the get-up timeout decides (limp)

    def __post_init__(self):
        self.state = State.LIMP
        self.command = Command()
        self._command_t = -1e9
        self._ramp_from: np.ndarray | None = None
        self._ramp_t0 = 0.0
        self.link_errors = 0
        self._link_down = False
        self._reconnect_t = -1e9
        self.link_ok_since = 0.0
        self.fall_events = 0          # falls that cut torque
        self.falls_seen = 0           # every fall verdict while running (keep mode gets up)
        self._was_fallen = False
        self._still_down = 0.0
        self._down_driving = 0.0      # down with the gait driving since the last hand-back
        self.reposes = 0              # re-poses since the robot was last upright
        self._run_t0 = 0.0
        self.refusal = ""
        self.stand_pose = DEFAULT_POSITION.copy()
        self.stand_pose[list(HEAD_JOINTS)] += np.asarray(self.head_offset, dtype=float)
        self._clear_moves()
        self.last_sensors: Sensors | None = None
        self.last_targets: np.ndarray | None = None
        self.predictor = FallPredictor()
        # Mouth (not a policy joint): held here, applied to every target. Moves only on a
        # build whose calibration gives the mouth a servo ID.
        self.mouth_target = MOUTH_CLOSED
        self.pick_mouth = True               # script the beak during a ground pick (if it has a servo)
        self._mouth_override: float | None = None

    def _clear_moves(self) -> None:
        self.move: _Move | None = None
        self.sit = "up"              # "up" | "sitting" | "rising"
        self.sit_settle = 0.0
        self.rise_left = 0.0
        self.mode = ""
        self._grace_until = -1e9
        self.fall_seq: dict | None = None     # on_fall="getup": the limp-fall sequence in progress

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
        lying = s.imu.gravity[2] > self.safety.fall_gravity_z
        if not s.imu_ready or (lying and self.on_fall == "limp"):
            # With keep / getup the policy may start from the floor: v8 gets up by itself.
            self.refusal = ("IMU not ready" if not s.imu_ready
                            else "robot is not upright (--keep-on-fall lets the policy get it up)")
            return False
        self.controller.reset()
        self._clear_moves()
        self._run_t0 = s.t
        self.reposes = 0                     # a fresh start gets the full set of re-poses
        self._still_down = self._down_driving = 0.0
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

    def swap_policies(self, policies, tuning=None) -> str | None:
        """Another policy set (official / kissqy...) with its gait tuning. Only while no network
        drives the body (LIMP / RAMP / HOLD): a running gait is never switched mid-step.
        None = done, else the refusal."""
        if self.state is State.RUN:
            return "stop the policy first (3), then switch"
        self.policies = policies
        self.controller.policy = policies.walk
        if tuning is not None:
            self.controller.tuning = tuning
        self.controller.reset()
        self._clear_moves()
        return None

    def set_command(self, command: Command, now: float) -> None:
        self.command = command
        self._command_t = now

    # -- one control period ----------------------------------------------------------------
    def _link_lost(self, err: Exception) -> None:
        """The body stopped answering (bridge reply lost, USB gone, servo feedback stale).
        The firmware drops torque by itself after 1 s of host silence; here: stop driving,
        say why, and try to reconnect once a second."""
        self.link_errors += 1
        self._link_down = True
        if self.state is not State.LIMP:
            self.state = State.LIMP
            self._clear_moves()
            self.refusal = f"link lost: {err}"
        now = time.monotonic()
        if hasattr(self.io, "reconnect") and now - self._reconnect_t >= 1.0:
            self._reconnect_t = now
            try:
                st = self.io.reconnect()
                why = {9: "brownout (supply dip)", 4: "crash", 1: "power-on", 3: "software reset",
                       5: "watchdog", 6: "watchdog", 7: "watchdog"}.get(st.get("reset_reason"), f"reason {st.get('reset_reason')}")
                self.refusal = (f"reconnected: bridge up {st.get('uptime_ms', 0) / 1000:.1f} s, last reset: {why}"
                                if isinstance(st, dict) else "reconnected")
                self.link_ok_since = now
                self._link_down = False
            except (RuntimeError, OSError) as e:
                self.refusal = f"link lost: {err}; reconnect failed: {e}"

    def tick(self) -> Sensors:
        self._mouth_override = None          # set again below while a ground pick runs
        try:
            s = self.io.read()
        except (RuntimeError, OSError) as e:
            self._link_lost(e)
            self.io.wait_tick()
            return self.last_sensors if self.last_sensors is not None else Sensors()
        if self._link_down:
            self._link_down = False
            self.refusal = "link restored (torque is off: enable again)"
        fallen = self.safety.update_fall(s) if s.imu_ready else False
        obs = targets = None

        if self.state is State.RUN and fallen and not self._was_fallen and self.fall_seq is None:
            self.falls_seen += 1
        if (self.state is State.RUN and self.on_fall == "getup" and self.fall_seq is None
                and s.imu_ready):
            # Catch the fall on the way down (robotd main.rs limp-fall): not for a robot already
            # down and working its way up, nor during a move or a sit (a roulade tips over on
            # purpose, a ground pick pitches forward, a sit is not a fall).
            if not fallen and not self.busy and self.sit != "sitting":
                if self.predictor.observe(s.imu, self.io.control_dt):
                    self.falls_seen += 1
                    self.io.set_torque(False)
                    self.fall_seq = {"phase": "limp", "since": s.t, "still": 0.0}
                    self.mode = "fall:limp"
            else:
                self.predictor.reset()
        if s.imu_ready and s.imu.gravity[2] < -0.9:          # upright (< ~25 deg): start over
            self.reposes = 0
        if not fallen:
            self._still_down = self._down_driving = 0.0
        elif (self.state is State.RUN and self.on_fall == "getup" and self.fall_seq is None
                and not (self.busy or self.sit == "sitting") and s.imu_ready):
            # Lying still, or getting nowhere: not getting up. Re-pose through the limp phase.
            dt = self.io.control_dt
            self._down_driving += dt
            if float(np.linalg.norm(s.imu.gyro)) < self.stuck_rate:
                self._still_down += dt
            else:
                self._still_down = 0.0
            stuck = (self._still_down >= self.stuck_repose_s - 1e-9
                     or self._down_driving >= self.stuck_noprogress_s - 1e-9)
            if stuck and self.reposes < self.max_reposes:
                self.reposes += 1
                self._still_down = self._down_driving = 0.0
                self.io.set_torque(False)
                self.fall_seq = {"phase": "limp", "since": s.t, "still": 0.0, "min": self.repose_limp_min_s}
                self.mode = "fall:limp"
        if (self.state is State.RUN and fallen and not (self.busy or self.sit == "sitting")
                and s.t >= self._grace_until and self.fall_seq is None):
            if self.on_fall == "limp":
                self.fall_events += 1
                self.limp()
            elif (self.getup_timeout_s is not None and self.safety._tilt_since is not None
                  and s.t - max(self.safety._tilt_since, self._run_t0) > self.getup_timeout_s):
                self.fall_events += 1
                self.refusal = f"still down after {self.getup_timeout_s:.0f} s: limp"
                self.limp()
        self._was_fallen = fallen and self.state is State.RUN

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
        elif self.state is State.RUN and self.fall_seq is not None:
            targets = self._limp_fall_step(s)
            obs = self.controller.last_obs if self.fall_seq is None else None
        elif self.state is State.RUN:
            targets = self._policy_step(s)
            obs = self.controller.last_obs

        self.last_sensors = s
        if targets is not None:
            targets[MOUTH_INDEX] = self.mouth_target if self._mouth_override is None else self._mouth_override
            targets = self.safety.clamp(targets)
            self.io.write(targets)
            self.last_targets = targets
        if self.log is not None:
            cur = getattr(self.io, "currents", None)
            self.log.append(TickLog(s.t, self.state.value, obs, None if targets is None else targets.copy(),
                                    s.positions.copy(), s.imu.gravity.copy(), s.imu.gyro.copy(), fallen,
                                    None if cur is None else np.array(cur, dtype=float),
                                    self.mode if self.state is State.RUN else "",
                                    None if getattr(self.io, "voltages", None) is None
                                    else np.array(self.io.voltages, dtype=float), self.on_fall))
        self.io.wait_tick()
        return s

    def _limp_fall_step(self, s: Sensors) -> np.ndarray | None:
        fs = self.fall_seq
        dt = self.io.control_dt
        if fs["phase"] == "limp":
            rate = float(np.linalg.norm(s.imu.gyro))
            fs["still"] = fs["still"] + dt if rate < self.limp_fall_still_rate else 0.0
            settled = fs["still"] >= self.limp_fall_still_s - 1e-9 and s.t - fs["since"] >= fs.get("min", 0.0) - 1e-9
            if settled or s.t - fs["since"] >= self.limp_fall_max_s:
                try:
                    self.io.set_torque(True)        # goal = where each joint lies: no jump
                except RuntimeError as e:
                    self.fall_events += 1
                    self.refusal = f"get-up: torque on refused: {e}"
                    self.limp()
                    return None
                self.fall_seq = {"phase": "pose", "since": s.t, "from": s.positions.copy()}
                self.mode = "fall:pose"
            return None
        a = (s.t - fs["since"]) / max(self.limp_fall_pose_s, 1e-6)
        if a >= 1.0:
            # Posed: hand back with fresh feedback state; the policy stands it up.
            self.fall_seq = None
            self.predictor.reset()
            self.controller.reset()
            self._run_t0 = s.t                      # get-up timeout counts from here
            self._down_driving = self._still_down = 0.0
            return self._policy_step(s)
        targets = fs["from"] + (self.stand_pose - fs["from"]) * a
        targets[MOUTH_INDEX] = MOUTH_CLOSED
        return targets

    def _policy_step(self, s: Sensors) -> np.ndarray:
        dt = self.io.control_dt
        # Expire windows first, so a tick after the deadline runs the next thing (control.rs).
        if self.move is not None and self.move.remaining <= 0:
            sk = self.move.skill
            if not self.move.unwinding and sk.unwind_s > 0:
                self.move = replace(self.move, unwinding=True, remaining=sk.unwind_s)
            else:
                self.move = None
                self._grace_until = s.t + self.post_move_grace_s
        if self.sit == "rising" and self.rise_left <= 0:
            self.sit = "up"

        cmd = self.safety.gate_command(self.command, s.t - self._command_t)
        policy = scale = lowpass = None
        if self.move is not None:
            # Head and body zeroed: every one-shot declares them unused (zero_command_padding).
            sk = self.move.skill
            if sk.encoding == "phase":
                ang = 2 * math.pi * self.move.elapsed / sk.period_s
                twist = (math.cos(ang), math.sin(ang), 0.0)
                if self.pick_mouth:
                    self._mouth_override = pick_mouth(self.move.elapsed / sk.period_s)
            else:
                twist = sk.unwind if self.move.unwinding else sk.twist
            cmd, policy, scale, lowpass = Command(twist=twist), sk.policy, sk.action_scale, sk.lowpass
            self.mode = sk.name + (":unwind" if self.move.unwinding else "")
        elif self.sit in ("sitting", "rising"):
            ss = self.policies.sitstand
            # The posture flag rides twist vx: 1 sit, 0 stand; head stays live.
            cmd = replace(cmd, twist=(1.0, 0.0, 0.0) if self.sit == "sitting" else (0.0, 0.0, 0.0))
            policy, scale, lowpass = ss.policy, ss.action_scale, ss.lowpass
            self.mode = "sit" if self.sit == "sitting" else "rise"
        else:
            self.mode = "walk"
        targets = self.controller.step(s, cmd, policy, scale, lowpass)

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
