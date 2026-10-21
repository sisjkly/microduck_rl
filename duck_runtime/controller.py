"""Sensors + command -> joint targets, with the official runtime's tuning semantics.

Mirrors robotd/src/control.rs:654-735 (pollen-robotics/microduck 942e58e) for the walk
net, which is the only net here (``velstand.onnx`` stands still at zero command, so no
separate stand network):

    obs      <- Observation::build(imu, q, qdot, DEFAULT_POSITION, last_action, cmd)
    action   <- ONNX                         (last_action = raw action, before scaling)
    targets  <- DEFAULT_POSITION + action_scale * scatter(action)
    filters  <- t = a*t + (1-a)*t_prev  on head (a=head_lowpass) and legs (a=legs_lowpass)

Official defaults (control.rs:70-79): action_scale 0.9, head_lowpass 0.5, legs_lowpass 0.7.
Training used scale 1.0 and no filter; both presets are available (``Tuning.official()``
/ ``Tuning.as_trained()``) and compared in sim before going to the robot.

Velocity: training lags joint_vel by exactly one control step (velocity cfg
``joint_vel.delay_min/max_lag = 1``). ``velocity_delay_ticks=1`` applies that lag here,
so both IO backends report instantaneous velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .contract import (
    ACTION_LEN, DEFAULT_POSITION, HEAD_JOINTS, MOUTH_CLOSED, MOUTH_INDEX, NUM_JOINTS, OBS_LEN,
    build_observation, command_block, scatter_action,
)
from .io import Sensors


@dataclass
class Tuning:
    action_scale: float = 0.9
    head_lowpass: float | None = 0.5
    legs_lowpass: float | None = 0.7
    velocity_delay_ticks: int = 1

    @classmethod
    def official(cls) -> "Tuning":
        return cls()

    @classmethod
    def as_trained(cls) -> "Tuning":
        return cls(action_scale=1.0, head_lowpass=None, legs_lowpass=None)


@dataclass
class Command:
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)   # vx, vy (m/s), wz (rad/s)
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    body_z: float = 0.0
    body_roll: float = 0.0
    body_pitch: float = 0.0

    def block(self) -> np.ndarray:
        return command_block(self.twist, self.head, self.body_z, self.body_roll, self.body_pitch)


class Policy:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if list(inp.shape)[-1] != OBS_LEN or list(out.shape)[-1] != ACTION_LEN:
            raise ValueError(f"{onnx_path}: expected [1,{OBS_LEN}] -> [1,{ACTION_LEN}], "
                             f"got {inp.shape} -> {out.shape}")
        self.inp, self.out = inp.name, out.name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = self.session.run([self.out], {self.inp: obs.reshape(1, OBS_LEN)})[0][0]
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("policy produced a non-finite action")
        return a.astype(np.float32)


class Controller:
    def __init__(self, policy: Policy, tuning: Tuning | None = None):
        self.policy = policy
        self.tuning = tuning or Tuning.official()
        self.reset()

    def reset(self) -> None:
        """control.rs:311-314: previous action zeroed, filter history dropped."""
        self.last_action = np.zeros(ACTION_LEN, dtype=np.float32)
        self.last_obs: np.ndarray | None = None
        self.previous_targets: np.ndarray | None = None
        self._vel_buffer: list[np.ndarray] = []

    def _delayed_velocity(self, v: np.ndarray) -> np.ndarray:
        k = self.tuning.velocity_delay_ticks
        if k <= 0:
            return v
        self._vel_buffer.append(v.copy())
        if len(self._vel_buffer) > k + 1:
            self._vel_buffer.pop(0)
        return self._vel_buffer[0]

    def observe(self, sensors: Sensors, command: Command) -> np.ndarray:
        return build_observation(
            sensors.imu.gyro, sensors.imu.gravity, sensors.positions,
            self._delayed_velocity(sensors.velocities), self.last_action, command.block(),
        )

    def step(self, sensors: Sensors, command: Command) -> np.ndarray:
        obs = self.observe(sensors, command)
        self.last_obs = obs
        action = self.policy(obs)
        self.last_action = action

        targets = DEFAULT_POSITION + self.tuning.action_scale * scatter_action(action)
        targets[MOUTH_INDEX] = MOUTH_CLOSED
        prev = self.previous_targets
        if prev is not None:
            a = self.tuning.head_lowpass
            if a is not None:
                for j in HEAD_JOINTS:
                    targets[j] = a * targets[j] + (1.0 - a) * prev[j]
            a = self.tuning.legs_lowpass
            if a is not None:
                for j in range(NUM_JOINTS):
                    if j in HEAD_JOINTS or j == MOUTH_INDEX:
                        continue
                    targets[j] = a * targets[j] + (1.0 - a) * prev[j]
        self.previous_targets = targets.copy()
        return targets
