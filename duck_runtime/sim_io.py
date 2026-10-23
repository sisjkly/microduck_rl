"""A simulated body behind the same RobotIO interface as the real one.

MuJoCo (CPU) + BAM M6 HD1910 actuator (STS3215 control law, firmware P=5), on the
microduck_rl MJCF patched with this build's mass profile (835 g, mouth fitted). The IMU is simulated
at the byte level: the trunk orientation/gyro are rotated into the board frame, packed
into the 12-byte SFLP block and decoded by the very ``SflpDecoder`` the robot uses — so
the mount quaternion and the decoder are exercised in sim too.

Not modelled: gear backlash, encoder bias, IMU noise, battery sag dynamics beyond BAM's
load drop. Same caveats as scripts/eval_mass_variants.py; mjlab_policy_eval.py is the
ground truth for policy quality, this is for the runtime's plumbing.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .contract import DEFAULT_POSITION, JOINT_NAMES, MOUTH_CLOSED, MOUTH_INDEX, NUM_JOINTS
from .imu import DEFAULT_MOUNT, SflpDecoder, encode_block, quat_mul, rotate_inverse
from .io import RobotIO, Sensors

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
HD1910_JSON = REPO / "src/mjlab_microduck/actuator/params/hd1910_m6.json"
STIFF_SOLREF = (-5.0e4, -2.0e2)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


class SimIO(RobotIO):
    def __init__(self, mass_profile: str | None = "ours_835g", servo: str = "hd1910",
                 kp_fw: float | None = None, vin: float | None = None, physics_dt: float = 0.005,
                 decimation: int = 4, actuator_delay_steps: int = 4, imu_mount=DEFAULT_MOUNT,
                 scene: Path = SCENE, seed: int = 0,
                 vin_drop_gain: float = 0.1, vin_min: float = 6.0):
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from bam.model import load_model
        from bam.mujoco import MujocoController
        from mjlab_microduck.robot.mass_profile import apply_mass_profile

        if servo == "hd1910":
            bam = load_model(str(HD1910_JSON))
            bam.actuator.kp = 5.0 if kp_fw is None else kp_fw
            bam.actuator.vin = 7.6 if vin is None else vin
        elif servo == "xl330":
            bam = load_model(motor_name="xl330", model="m6")
            bam.actuator.kp = 200.0 if kp_fw is None else kp_fw
            bam.actuator.vin = 7.4 if vin is None else vin
        else:
            raise ValueError(servo)
        bam.actuator.max_current = None
        force = bam.actuator.vin * bam.kt.value / bam.R.value

        spec = mujoco.MjSpec.from_file(str(scene))
        apply_mass_profile(spec, mass_profile)
        names = []
        for act in spec.actuators:
            tgt = act.target if isinstance(act.target, str) else act.target.name
            if tgt.startswith("passive_"):
                continue
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force, force)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            names.append(act.name)
            j = spec.joint(tgt)
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = STIFF_SOLREF
            j.solimp_friction = STIFF_SOLIMP
        self.model = spec.compile()
        self.model.opt.timestep = physics_dt
        self.data = mujoco.MjData(self.model)
        self.bam = bam
        # Supply sag: V = vin - vin_drop_gain * sum|tau| (V per N.m), floored at vin_min.
        # Training: gain 0-0.2, floor 6 V (XL330). This robot (set6.npz): 7.5 V idle, 4.6 V at the
        # rise's peak and 3-4 V in a kick -> --vin-min well below 6 reproduces it.
        self.ctrl = MujocoController(bam, names, self.model, self.data, vin_drop_gain=vin_drop_gain, vin_min=vin_min)
        self.voltages = np.full(NUM_JOINTS, float(bam.actuator.vin))
        # Our 15-joint order -> model actuator order (the sim has no mouth joint).
        self._sim_joints = [j for j in range(NUM_JOINTS) if j != MOUTH_INDEX]
        self._act_index = [names.index(JOINT_NAMES[j]) for j in self._sim_joints]
        jids = [self.model.joint(JOINT_NAMES[j]).id for j in self._sim_joints]
        self.qadr = self.model.jnt_qposadr[jids]
        self.vadr = self.model.jnt_dofadr[jids]
        self.trunk = self.model.body("trunk_base").id
        self.gyro_adr = self.model.sensor_adr[self.model.sensor("imu_ang_vel").id]
        self.decimation = decimation
        self.control_dt = physics_dt * decimation
        self.delay = actuator_delay_steps
        self.mount = np.asarray(imu_mount, dtype=float)
        self.decoder = SflpDecoder(imu_mount)
        self.rng = np.random.default_rng(seed)
        self.torque = False
        self.reset()

    # -- sim-only helpers ------------------------------------------------------------
    def reset(self, joint_noise: float = 0.0, z: float = 0.125) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, z)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = DEFAULT_POSITION[self._sim_joints] + self.rng.uniform(-joint_noise, joint_noise, 14)
        mujoco.mj_forward(m, d)
        self.ctrl.reset(d.qpos)
        self.ctrl.last_ts = d.time
        if type(self.bam.actuator).__name__ == "STS3215Actuator":
            self.bam.actuator.q_target_smooth = d.qpos[self.qadr].copy()
        self._queue = [d.qpos[self.qadr].copy()] * max(self.delay, 1)
        self._target = d.qpos[self.qadr].copy()
        self._mouth = MOUTH_CLOSED
        self.decoder = SflpDecoder(self.mount)
        # Let SFLP "warm up" like the real chip (decoder ready after 25 samples).
        for _ in range(SflpDecoder.READY_SAMPLES):
            self.decoder.decode(self._imu_block())

    def push(self, dvx: float, dvy: float) -> None:
        self.data.qvel[0] += dvx
        self.data.qvel[1] += dvy

    def trunk_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def trunk_quat(self) -> np.ndarray:
        return self.data.xquat[self.trunk].copy()

    # -- RobotIO -------------------------------------------------------------------------
    def _imu_block(self) -> bytes:
        q_trunk = self.data.xquat[self.trunk]
        gyro_trunk = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        q_sensor = quat_mul(q_trunk, self.mount)
        gyro_sensor = rotate_inverse(self.mount, gyro_trunk)
        return encode_block(gyro_sensor, q_sensor)

    def read(self) -> Sensors:
        d = self.data
        pos = np.zeros(NUM_JOINTS)
        vel = np.zeros(NUM_JOINTS)
        pos[self._sim_joints] = d.qpos[self.qadr]
        vel[self._sim_joints] = d.qvel[self.vadr]
        pos[MOUTH_INDEX] = self._mouth
        imu = self.decoder.decode(self._imu_block())
        return Sensors(positions=pos, velocities=vel, imu=imu, imu_ready=self.decoder.ready, t=d.time)

    def now(self) -> float:
        return self.data.time

    def write(self, targets: np.ndarray) -> None:
        self._target = np.asarray(targets, dtype=float)[self._sim_joints].copy()
        self._mouth = float(targets[MOUTH_INDEX])

    def set_torque(self, on: bool) -> None:
        if on and not self.torque:
            # A powered servo's internal goal starts where the horn is (goal reset).
            q = self.data.qpos[self.qadr].copy()
            self._queue = [q.copy()] * max(self.delay, 1)
            self._target = q
            self.ctrl.last_ts = self.data.time
            if type(self.bam.actuator).__name__ == "STS3215Actuator":
                self.bam.actuator.q_target_smooth = q.copy()
        self.torque = on

    def wait_tick(self) -> None:
        for _ in range(self.decimation):
            self._queue.append(self._target.copy())
            q_target = self._queue.pop(0)
            if self.torque:
                self.ctrl.q_target[self._act_index] = q_target
                self.ctrl.update()
                c = self.ctrl
                if c.vin_drop_gain is not None:
                    v = self.bam.actuator.vin - c.vin_drop_gain * float(np.sum(np.abs(c._prev_motor_torque)))
                    self.voltages[:] = max(v, c.vin_min) if c.vin_min is not None else v
            else:
                self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
