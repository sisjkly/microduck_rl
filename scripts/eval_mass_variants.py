#!/usr/bin/env python3
"""Headless CPU battery: one ONNX policy x {mass profile} x {servo model}.

Answers "does this policy still work on OUR robot?" before touching hardware.
Same plant as scripts/infer_policy.py (BAM on a motor-converted MJCF, CPU
MuJoCo, 200 Hz physics / 50 Hz policy, 61-D obs) with the variables we care
about exposed:

  --profiles   mass profiles from src/mjlab_microduck/robot/mass_profile.py
  --servos     xl330  : BAM M6 XL330, kp_fw 200 (what official policies train on)
               hd1910 : BAM M6 fit of the Feetech HD1910 (kissqy / microduck-replica
                        json), STS3215 control law, firmware P=5 (kissqy runtime)

Scenarios (each over --seeds seeds, small random initial joint noise):
  stand      10 s zero command
  walk_*     10 s constant twist (fwd / back / side / turn)
  push_*     stand, horizontal velocity kick of the given size at t=3 s
A run "falls" if trunk tilt exceeds 60 deg or trunk z drops below 6 cm.

Not modelled (kept identical across configs, so A/B deltas are meaningful):
gear backlash, IMU noise/misalignment, encoder bias, battery-voltage DR.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mjlab_microduck.robot.mass_profile import apply_mass_profile  # noqa: E402

SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
HD1910_JSON_DEFAULT = REPO / "src/mjlab_microduck/actuator/params/hd1910_m6.json"

JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                 0.0, 0.0873, 0.4579, 0.0049, -0.4530])
SERVOS = {
    "xl330": dict(motor_name="xl330", kp_fw=200.0, vin=7.4),
    "hd1910": dict(json=None, kp_fw=5.0, vin=7.6),
}
STIFF_SOLREF = (-5.0e4, -2.0e2)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)
DT, DECIM = 0.005, 4
ACT_DELAY_STEPS = 4  # 20 ms, middle of training's 3-6 physics-step lag
# Actor obs noise exactly as microduck_velocity_env_cfg (uniform +-n): gyro, gravity,
# joint_pos, joint_vel. Training never shows the policy a noiseless world.
OBS_NOISE = np.concatenate([[0.03] * 3, [0.01] * 3, [0.001] * 14, [0.25] * 14, [0.0] * 14, [0.0] * 13])
USE_OBS_NOISE = True

SCENARIOS = {
    "stand":     dict(cmd=(0.0, 0.0, 0.0)),
    "walk_fwd":  dict(cmd=(0.3, 0.0, 0.0)),
    "walk_back": dict(cmd=(-0.3, 0.0, 0.0)),
    "walk_side": dict(cmd=(0.0, 0.2, 0.0)),
    "turn":      dict(cmd=(0.0, 0.0, 1.0)),
    "push_0.3":  dict(cmd=(0.0, 0.0, 0.0), push=0.3),
    "push_0.5":  dict(cmd=(0.0, 0.0, 0.0), push=0.5),
    "push_0.7":  dict(cmd=(0.0, 0.0, 0.0), push=0.7),
}


def build(profile: str, servo: str, hd1910_json: Path):
    from bam.model import load_model
    from bam.mujoco import MujocoController

    cfg = SERVOS[servo]
    bam = load_model(motor_name=cfg["motor_name"], model="m6") if servo == "xl330" else load_model(str(hd1910_json))
    bam.actuator.kp = cfg["kp_fw"]
    bam.actuator.vin = cfg["vin"]
    bam.actuator.max_current = None
    force_limit = bam.actuator.vin * bam.kt.value / bam.R.value

    spec = mujoco.MjSpec.from_file(str(SCENE))
    apply_mass_profile(spec, profile)
    names = []
    for act in spec.actuators:
        tgt = act.target if isinstance(act.target, str) else act.target.name
        if tgt.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        j = spec.joint(tgt)
        j.damping = np.zeros((3, 1))
        j.frictionloss = 0.0
        j.solref_friction = STIFF_SOLREF
        j.solimp_friction = STIFF_SOLIMP
    assert names == JOINTS, names
    model = spec.compile()
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    ctrl = MujocoController(bam, names, model, data, vin_drop_gain=0.1, vin_min=6.0)
    return model, data, ctrl, bam


class Runner:
    def __init__(self, onnx_path: str, profile: str, servo: str, hd1910_json: Path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.model, self.data, self.ctrl, self.bam = build(profile, servo, hd1910_json)
        m = self.model
        jid = [m.joint(n).id for n in JOINTS]
        self.qadr = m.jnt_qposadr[jid]
        self.vadr = m.jnt_dofadr[jid]
        self.trunk = m.body("trunk_base").id
        self.gyro_adr = m.sensor_adr[m.sensor("imu_ang_vel").id]
        self.stall = self.bam.actuator.vin * self.bam.kt.value / self.bam.R.value

    def reset(self, rng):
        self.rng = rng
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.125)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = HOME + rng.uniform(-0.02, 0.02, size=14)
        mujoco.mj_forward(m, d)
        self.ctrl.reset(d.qpos)
        self.ctrl.last_ts = d.time  # MujocoController keeps last_ts across mj_resetData -> dt<0 on the
        # first update, which corrupts the STS3215 law's velocity-clamped target (dt-dependent).
        self.ctrl.q_target[:] = HOME
        if type(self.bam.actuator).__name__ == "STS3215Actuator":  # goal reset (kissqy Hd1910Actuator)
            self.bam.actuator.q_target_smooth = d.qpos[self.qadr].copy()
        self.last_action = np.zeros(14, dtype=np.float32)
        self.prev_qvel = d.qvel[self.vadr].copy()  # 1-ctrl-step joint_vel lag
        self.target_queue = [HOME.copy()] * ACT_DELAY_STEPS

    def obs(self, cmd13):
        d = self.data
        q = d.xquat[self.trunk]
        g = np.zeros(3)
        mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), np.array([q[0], -q[1], -q[2], -q[3]]))
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        jp = d.qpos[self.qadr] - HOME
        o = np.concatenate([gyro, g, jp, self.prev_qvel, self.last_action, cmd13])
        if USE_OBS_NOISE:
            o = o + self.rng.uniform(-1.0, 1.0, size=o.shape) * OBS_NOISE
        self.prev_qvel = d.qvel[self.vadr].copy()
        return o.astype(np.float32)

    def tilt_deg(self):
        q = self.data.xquat[self.trunk]
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))

    def run(self, scen: dict, seed: int, seconds: float = 10.0):
        rng = np.random.default_rng(seed)
        self.reset(rng)
        m, d = self.model, self.data
        cmd13 = np.zeros(13, dtype=np.float32)
        cmd13[0:3] = scen["cmd"]
        push_t = 3.0 if "push" in scen else None
        push_dir = rng.uniform(0, 2 * math.pi)
        n_ctrl = int(seconds / (DT * DECIM))
        fell_at, tilts, xy0, sat, head_err = None, [], d.qpos[0:2].copy(), [], []
        yaw0 = None
        for k in range(n_ctrl):
            t = k * DT * DECIM
            if push_t is not None and abs(t - push_t) < 1e-9:
                d.qvel[0] += scen["push"] * math.cos(push_dir)
                d.qvel[1] += scen["push"] * math.sin(push_dir)
            a = self.sess.run(None, {self.inp: self.obs(cmd13)[None]})[0][0]
            self.last_action = a.astype(np.float32)
            target = HOME + a
            for _ in range(DECIM):
                self.target_queue.append(target)
                self.ctrl.q_target[:] = self.target_queue.pop(0)
                self.ctrl.update()
                mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                fell_at = t
                break
            tl = self.tilt_deg()
            tilts.append(tl)
            sat.append(np.abs(d.ctrl).max() / self.stall)
            head_err.append(np.degrees(d.qpos[self.qadr[5:7]] - HOME[5:7]))
            if t >= 2.0 and yaw0 is None:
                yaw0, xy_start, t_start = self._yaw(), d.qpos[0:2].copy(), t
                yaw_prev, yaw_acc = yaw0, 0.0
            elif yaw0 is not None:
                y = self._yaw()
                yaw_acc += _wrap(y - yaw_prev)
                yaw_prev = y
            if fell_at is None and (tl > 60.0 or d.qpos[2] < 0.06):
                fell_at = t
                break
        out = dict(fell=fell_at is not None, fell_at=fell_at,
                   tilt_mean=float(np.mean(tilts[50:])) if len(tilts) > 50 else float("nan"),
                   tilt_max=float(np.max(tilts)) if tilts else float("nan"),
                   sat_p95=float(np.percentile(sat, 95)) if sat else float("nan"),
                   head_pitch_err=np.mean(head_err[50:], axis=0).round(1).tolist() if len(head_err) > 50 else None)
        if fell_at is None and yaw0 is not None:
            T = seconds - DT * DECIM - t_start
            dxy = d.qpos[0:2] - xy_start
            c, s = math.cos(yaw0), math.sin(yaw0)
            out["vx"] = float((c * dxy[0] + s * dxy[1]) / T)
            out["vy"] = float((-s * dxy[0] + c * dxy[1]) / T)
            out["wz"] = float(yaw_acc / T)
        return out

    def _yaw(self):
        q = self.data.xquat[self.trunk]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--profiles", nargs="+", default=["official", "ours_813g", "ours_813g_headfront"])
    ap.add_argument("--servos", nargs="+", default=["xl330", "hd1910"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--hd1910-json", type=Path, default=HD1910_JSON_DEFAULT)
    ap.add_argument("--hd1910-kp", type=float, default=None, help="override HD1910 firmware P")
    ap.add_argument("--no-obs-noise", action="store_true")
    ap.add_argument("--act-delay", type=int, default=ACT_DELAY_STEPS, help="actuator lag in 5 ms physics steps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    globals()["ACT_DELAY_STEPS"] = args.act_delay
    globals()["USE_OBS_NOISE"] = not args.no_obs_noise
    if args.hd1910_kp is not None:
        SERVOS["hd1910"]["kp_fw"] = args.hd1910_kp

    results = {}
    for profile, servo in itertools.product(args.profiles, args.servos):
        r = Runner(args.onnx, profile, servo, args.hd1910_json)
        for sc in args.scenarios:
            runs = [r.run(SCENARIOS[sc], seed) for seed in range(args.seeds)]
            results[(profile, servo, sc)] = runs
            ok = [x for x in runs if not x["fell"]]
            line = f"{profile:20s} {servo:7s} {sc:10s} fall {sum(x['fell'] for x in runs)}/{len(runs)}"
            if ok:
                line += f"  tilt {np.mean([x['tilt_mean'] for x in ok]):4.1f}°(max {np.max([x['tilt_max'] for x in ok]):4.1f})"
                line += f"  sat95 {np.mean([x['sat_p95'] for x in ok]):.2f}"
                if "vx" in ok[0]:
                    line += f"  v=({np.mean([x['vx'] for x in ok]):+.2f},{np.mean([x['vy'] for x in ok]):+.2f},{np.mean([x['wz'] for x in ok]):+.2f})"
                line += f"  head/neck pitch err {np.mean([x['head_pitch_err'] for x in ok], axis=0).round(1).tolist()}°"
            print(line, flush=True)
    if args.out:
        args.out.write_text(json.dumps({"|".join(k): v for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
