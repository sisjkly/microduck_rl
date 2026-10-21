#!/usr/bin/env python3
"""Run an exported ONNX policy inside the REAL mjlab training env (warp, CPU ok).

Ground truth for the CPU rehearsal tools: same env cfg the policy was trained
in (obs pipeline + noise, BAM actuator incl. delay, contacts, DR), with pushes
removed and commands pinned. Mass profile via MICRODUCK_MASS_PROFILE, servo
model via --servo (xl330 | hd1910).

  MICRODUCK_MASS_PROFILE=official uv run python scripts/mjlab_policy_eval.py \
      Mjlab-VelStand-Flat-MicroDuck velstand.onnx --cmds "0.3,0,0;-0.3,0,0"
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os

import numpy as np
import onnxruntime as ort
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("onnx")
    ap.add_argument("--cmds", default="0,0,0;0.1,0,0;0.3,0,0;-0.3,0,0;0,0.2,0;0,0,1.0",
                    help="';'-separated vx,vy,wz triples")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--servo", choices=["xl330", "hd1910"], default="xl330")
    ap.add_argument("--keep-pushes", action="store_true")
    ap.add_argument("--device", default="cpu")
    # Runtime post-processing (duck_runtime / official robotd control.rs semantics):
    # target = HOME + scale * a, then first-order low-pass on head / legs targets; the
    # policy's "previous action" obs slot gets the RAW action, as the official runtime does.
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--head-lowpass", type=float, default=None)
    ap.add_argument("--legs-lowpass", type=float, default=None)
    args = ap.parse_args()

    # Servo model is chosen in microduck_constants at import time.
    os.environ["MICRODUCK_SERVO"] = args.servo
    import mjlab_microduck.tasks  # noqa: F401  registers tasks (and mdp patches)
    from mjlab_microduck.robot import microduck_constants as mc
    assert mc.SERVO_MODEL == args.servo
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(args.task, play=False)
    cfg.scene.num_envs = args.num_envs
    if not args.keep_pushes:
        for ev in ("push_robot", "topple_push"):
            cfg.events.pop(ev, None)
    tw = cfg.commands["twist"]
    tw.resampling_time_range = (1e6, 1e6)
    tw.rel_standing_envs = 0.0
    if hasattr(tw, "rel_turn_in_place_envs"):
        tw.rel_turn_in_place_envs = 0.0
    for name in ("head_pose", "body_pose"):
        if name in cfg.commands:
            c = cfg.commands[name]
            c.resampling_time_range = (1e6, 1e6)
            c.ranges = tuple((0.0, 0.0) for _ in c.ranges)
    # Freeze every curriculum at its current (final-stage for warm-started tasks
    # is irrelevant here: no reward is used) — curricula only touch DR ranges.
    cfg.curriculum = {}
    # No episode timeouts / terminations mid-measure except falls.
    cfg.episode_length_s = args.seconds + 2.0

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    robot = env.scene["robot"]
    mass = float(env.sim.mj_model.body_mass.sum())
    print(f"[mjlab] task={args.task} servo={args.servo} mass={mass*1000:.1f} g profile={os.environ.get('MICRODUCK_MASS_PROFILE','(default)')} "
          f"scale={args.action_scale} head_lp={args.head_lowpass} legs_lp={args.legs_lowpass}")

    n_steps = int(args.seconds / env.step_dt)
    for cs in args.cmds.split(";"):
        cmd = torch.tensor([float(x) for x in cs.split(",")], device=env.device)
        obs, _ = env.reset()
        term = env.command_manager.get_term("twist")
        fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        vsum = torch.zeros(env.num_envs, 3, device=env.device)
        cnt = 0
        raw_prev = np.zeros((env.num_envs, 14), dtype=np.float32)
        off_prev = None
        head = np.zeros(14, dtype=bool)
        head[5:9] = True  # policy slots 5..8 = neck_pitch, head_pitch, head_yaw, head_roll
        for k in range(n_steps):
            term.vel_command_b[:] = cmd
            term.is_standing_env[:] = False
            ob = obs["actor"].cpu().numpy().astype(np.float32)
            ob[:, 34:48] = raw_prev
            a = np.concatenate([sess.run(None, {inp: ob[i:i + 1]})[0] for i in range(ob.shape[0])])
            raw_prev = a.copy()
            off = args.action_scale * a
            if off_prev is not None:
                if args.head_lowpass is not None:
                    off[:, head] = args.head_lowpass * off[:, head] + (1 - args.head_lowpass) * off_prev[:, head]
                if args.legs_lowpass is not None:
                    off[:, ~head] = args.legs_lowpass * off[:, ~head] + (1 - args.legs_lowpass) * off_prev[:, ~head]
            obs, rew, terminated, truncated, info = env.step(torch.as_tensor(off, device=env.device))
            done = (terminated | truncated).cpu().numpy()
            off_prev = off.copy()
            off_prev[done] = 0.0
            raw_prev[done] = 0.0
            q = robot.data.root_link_quat_w
            tilt = torch.rad2deg(torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)))
            fell |= (tilt > 60) | terminated
            if k * env.step_dt >= 2.0:
                v = robot.data.root_link_lin_vel_b
                w = robot.data.root_link_ang_vel_b
                vsum += torch.stack([v[:, 0], v[:, 1], w[:, 2]], dim=1)
                cnt += 1
        ok = ~fell
        vm = (vsum[ok] / max(cnt, 1)).mean(0).tolist() if ok.any() else [math.nan] * 3
        print(f"  cmd {cs:>10s}: fall {int(fell.sum())}/{env.num_envs}  v=({vm[0]:+.2f},{vm[1]:+.2f},{vm[2]:+.2f})", flush=True)


if __name__ == "__main__":
    main()
