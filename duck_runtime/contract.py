"""The policy contract, mirrored from the official runtime (pollen-robotics/microduck 942e58e).

Every number here has a source line in the official repo; a mismatch does not fail loudly,
it produces a robot that falls over. ``tests/test_duck_runtime.py`` pins them against
microduck_rl's training cfg (HOME_FRAME, joint order) so the two cannot drift.

  joint names / ids ...... duck-ipc-proto/src/lib.rs:572-600
  DEFAULT_POSITION ....... duck-control/src/model.rs:37-53   (== training HOME_FRAME)
  MOUTH_INDEX ............ duck-control/src/model.rs:29
  observation layout ..... duck-control/src/obs.rs:12-33, build() 174-231
  scatter / gather ....... duck-control/src/obs.rs:233-256
"""

from __future__ import annotations

import math

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
NUM_JOINTS = 15
# Bus IDs, indexed as JOINT_NAMES. Same table on the official (Dynamixel) robot and on
# kissqy's HD1910 build; the IMU board answers as 200 on the same bus.
JOINT_IDS: tuple[int, ...] = (20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14)
IMU_ID = 200
MOUTH_INDEX = 9
HEAD_JOINTS = range(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll (low-pass group)

DEFAULT_POSITION = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # left leg
    0.3491, 0.3491, 0.0, 0.0,                 # neck_pitch, head_pitch, head_yaw, head_roll
    0.0,                                      # mouth
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,     # right leg
])
MOUTH_CLOSED = -5.0 * math.pi / 180.0

# Joint travel from the MJCF (robot_walk.xml), used as the default clamp. The mouth range
# is kissqy's measured one. Calibration files may only narrow these.
JOINT_LIMITS = np.array([
    (-0.4363, 0.5236), (-0.3840, 0.3840), (-1.5708, 1.5708), (-1.5708, 1.5708), (-1.5708, 1.5708),
    (-1.5708, 1.0472), (-1.5708, 1.5708), (-2.9671, 2.9671), (-0.4363, 0.4363),
    (-0.0873, 0.5236),
    (-0.5236, 0.4363), (-0.3840, 0.3840), (-1.5708, 1.5708), (-1.5708, 1.5708), (-1.5708, 1.5708),
])

OBS_LEN = 61
ACTION_LEN = 14
COMMAND_LEN = 13

# Policy slot -> joint index: below the mouth straight through, at/above shifted by one.
POLICY_JOINTS = np.array([s if s < MOUTH_INDEX else s + 1 for s in range(ACTION_LEN)])


def policy_joints(values: np.ndarray) -> np.ndarray:
    """15 joints -> the 14 a policy sees (mouth removed)."""
    return np.asarray(values, dtype=np.float64)[POLICY_JOINTS]


def scatter_action(action: np.ndarray) -> np.ndarray:
    """14 policy outputs -> 15 joints, mouth slot left at 0."""
    out = np.zeros(NUM_JOINTS)
    out[POLICY_JOINTS] = action
    return out


def command_block(twist=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0, 0.0),
                  body_z=0.0, body_roll=0.0, body_pitch=0.0) -> np.ndarray:
    """The 13-wide command. Body x, y, yaw are unbound in training -> always 0.
    Note the body order: z, roll, pitch (obs.rs:30-33)."""
    return np.array([*twist, *head, 0.0, 0.0, body_z, body_roll, body_pitch, 0.0], dtype=np.float64)


def build_observation(gyro, gravity, joint_positions, joint_velocities, last_action, command) -> np.ndarray:
    """obs.rs build(): gyro | gravity | q - home | qdot | previous raw action | command."""
    obs = np.empty(OBS_LEN, dtype=np.float32)
    obs[0:3] = gyro
    obs[3:6] = gravity
    obs[6:20] = policy_joints(joint_positions) - policy_joints(DEFAULT_POSITION)
    obs[20:34] = policy_joints(joint_velocities)
    obs[34:48] = last_action
    obs[48:61] = command
    return obs
