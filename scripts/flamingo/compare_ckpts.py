"""One-line battery per checkpoint, to pick the one to deploy (CPU).

    uv run python scripts/flamingo/compare_ckpts.py logs/rsl_rl/microduck_flamingo/<run>/model_{1000,2000,3000}.pt
    uv run python scripts/flamingo/compare_ckpts.py <ckpts...> --push 0 0.15 --num-envs 16

Each run: spawn HOME, two feet for 1 s, then one foot (sides alternate across envs), 8 s.
Over the last 3 s it reports:
  ok               alive and >= 95 % single support on the commanded side
  tilt             trunk tilt, median (deg)
  lift_mm          swing foot height, median over OK envs
  over_lim_max_rad worst commanded target past a joint limit (HOME + action vs MJCF range)
  joints_over_lim  joints whose mean over-limit exceeds 0.02 rad (parked on a hard stop)
  tau_p95_max      per-joint p95 |torque|, worst joint (N·m) — servo heat / stall risk
  dact             mean |delta action| per step (jitter)
"""

from __future__ import annotations

import argparse

import torch

from _common import TASK, alternating_sides, load_policy, make_env, set_flamingo_cmd
from mjlab_microduck.tasks import mdp as M


def battery(ck, push, N=32, T=8.0, seed=0, task=TASK):
    torch.manual_seed(seed)
    env = make_env(N, 0.0, push, task)
    wrapped, pol = load_policy(env, ck, task)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    ids = M._servo_joint_ids(env, robot)
    lim = robot.data.joint_pos_limits[0, ids]
    home = M._servo_default_joint_pos(env, robot)[0]
    side = alternating_sides(N)
    dt = env.step_dt
    steps, win = int(T / dt), int(3.0 / dt)
    alive = torch.ones(N, dtype=torch.bool)
    one, tilt, over, tau, dact, lift = [], [], [], [], [], []
    prev = None
    for k in range(steps):
        set_flamingo_cmd(env, 1.0 if k * dt >= 1.0 else 0.0, side)
        with torch.no_grad():
            a = pol(obs)
        obs, _, d, _ = wrapped.step(a)
        alive &= ~d.bool()
        if k >= steps - win:
            fd = M._flamingo_feet_found(env, "feet_ground_contact")
            sr = side > 0
            one.append((torch.where(sr, fd[:, 1], fd[:, 0]) & ~torch.where(sr, fd[:, 0], fd[:, 1])).float())
            g = robot.data.projected_gravity_b
            tilt.append(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))))
            tgt = home + a
            over.append(torch.relu(tgt - lim[:, 1]) + torch.relu(lim[:, 0] - tgt))
            tau.append(robot.data.actuator_force.abs())
            hh = M._flamingo_foot_heights(env, robot)
            lift.append(torch.where(sr, hh[:, 0], hh[:, 1]))
            if prev is not None:
                dact.append((a - prev).abs().mean(1))
        prev = a.clone()
    ok = alive & (torch.stack(one).mean(0) > 0.95)
    over = torch.stack(over)
    res = {
        "ok": f"{int(ok.sum())}/{N}",
        "alive": f"{int(alive.sum())}/{N}",
        "tilt": round(float(torch.stack(tilt).median()), 1),
        "lift_mm": round(float(torch.stack(lift)[:, ok].median()) * 1000) if ok.any() else None,
        "over_lim_max_rad": round(float(over.amax(dim=(0, 1)).max()), 2),
        "joints_over_lim": int((over.mean(dim=(0, 1)) > 0.02).sum()),
        "tau_p95_max": round(float(torch.stack(tau).quantile(0.95, dim=0).quantile(0.95, dim=0).max()), 2),
        "dact": round(float(torch.stack(dact).mean()), 3),
    }
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--push", type=float, nargs="+", default=[0.0, 0.15])
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    for ck in a.ckpts:
        for p in a.push:
            print(ck.split("/")[-1], f"push {p}:", battery(ck, p, N=a.num_envs, task=a.task), flush=True)


if __name__ == "__main__":
    main()
