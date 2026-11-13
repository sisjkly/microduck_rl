"""Closed-loop scenarios for ONE flamingo checkpoint (CPU), scenario by scenario.

    uv run python scripts/flamingo/eval_scenarios.py logs/rsl_rl/microduck_flamingo/<run>/model_1000.pt
    uv run python scripts/flamingo/eval_scenarios.py <ckpt> --scenario lift hold --push 0.15

Scenarios (N envs each, command forced every step):
  hold : spawn on one foot (side from the spawn), command one foot on that side
  lift : spawn HOME, two feet for 1 s, then one foot (side alternates across envs)
  down : spawn on one foot, one foot for 1 s, then two feet
  idle : spawn HOME, two feet
Metrics over the last 3 s: success, per-side success, swing height / distance behind,
trunk tilt, swing-foot touches, lift-off time, action magnitude and jitter.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import TASK, alternating_sides, load_policy, make_env, set_flamingo_cmd
from mjlab_microduck.tasks import mdp as M

SCENARIOS = ("hold", "lift", "down", "idle")


def run(ckpt, scenario, N=32, T=8.0, push=0.0, seed=0, task=TASK):
    torch.manual_seed(seed)
    spawn = 1.0 if scenario in ("hold", "down") else 0.0
    env = make_env(N, spawn, push, task)
    wrapped, policy = load_policy(env, ckpt, task)
    obs = wrapped.get_observations()
    term = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    dt = env.step_dt
    alive = torch.ones(N, dtype=torch.bool)
    side = torch.ones(N)
    rec = {k: [] for k in ("both", "sup", "sw", "lift", "behind", "tilt", "act_abs", "dact")}
    prev_a = None
    for k in range(int(T / dt)):
        t = k * dt
        if k == 1:
            if scenario in ("hold", "down"):
                h = M._flamingo_foot_heights(env, robot)
                side = torch.where(h[:, 1] < h[:, 0], 1.0, -1.0)    # right foot lower → right support
            else:
                side = alternating_sides(N)
        flag = {"hold": 1.0, "lift": float(t >= 1.0), "down": float(t < 1.0), "idle": 0.0}[scenario]
        set_flamingo_cmd(env, flag, side)
        with torch.no_grad():
            a = policy(obs)
        obs, _, dones, _ = wrapped.step(a)
        alive &= ~dones.bool()
        f = M._flamingo_feet_found(env, "feet_ground_contact")
        h = M._flamingo_foot_heights(env, robot)
        sr = side > 0
        rec["sup"].append(torch.where(sr, f[:, 1], f[:, 0]).clone())
        rec["sw"].append(torch.where(sr, f[:, 0], f[:, 1]).clone())
        rec["both"].append((f[:, 0] & f[:, 1]).clone())
        rec["lift"].append(torch.where(sr, h[:, 0], h[:, 1]).clone())
        term._active_side[:] = side                     # behind-distance reads the active side
        rec["behind"].append(M.flamingo_swing_behind_dist(env, "twist", robot).clone())
        g = robot.data.projected_gravity_b
        rec["tilt"].append(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).clone())
        rec["act_abs"].append(a.abs().max(dim=1).values.clone())
        rec["dact"].append((a - prev_a).abs().mean(dim=1).clone() if prev_a is not None else torch.zeros(N))
        prev_a = a.clone()
        if (~alive).all():
            break
    R = {k: torch.stack(v) for k, v in rec.items()}
    env.close()
    return R, alive, side, dt


def summarize(R, alive, side, dt, scenario, window_s=3.0):
    n = int(window_s / dt)
    W = {k: v[-n:].float() for k, v in R.items()}
    one = (W["sup"] > 0) & ~(W["sw"] > 0)
    two = W["both"] > 0
    out = {"survived": f"{int(alive.sum())}/{len(alive)}"}
    if scenario in ("hold", "lift"):
        ok = alive & (one.float().mean(0) > 0.95)
        out["one-foot OK (last 3 s >= 95 % single support)"] = f"{int(ok.sum())}/{len(alive)}"
        for s, nm in ((1, "R"), (-1, "L")):
            m = side == s
            out[f"  side {nm} OK"] = f"{int((ok & m).sum())}/{int(m.sum())}"
        if ok.any():
            out["lift mm (p50, OK envs)"] = round(float(W["lift"][:, ok].median()) * 1000, 1)
            out["behind mm (p50, OK envs)"] = round(float(W["behind"][:, ok].median()) * 1000, 1)
            out["tilt deg (p50 / p95)"] = (round(float(W["tilt"][:, ok].median()), 1),
                                           round(float(W["tilt"][:, ok].quantile(0.95)), 1))
            out["swing touches (share of steps)"] = round(float((W["sw"][:, ok] > 0).float().mean()), 3)
        if scenario == "lift":
            first = []
            for i in range(len(alive)):
                o = ((R["sup"][:, i] > 0) & ~(R["sw"][:, i] > 0)).float().numpy()
                idx = np.where(o[int(1.0 / dt):] > 0)[0]
                first.append(idx[0] * dt if len(idx) else np.nan)
            first = np.array(first)
            lifted = np.isfinite(first).any()
            out["lift-off s after command (p50)"] = round(float(np.nanmedian(first)), 2) if lifted else None
            out["never lifted"] = int(np.isnan(first).sum())
    else:
        ok = alive & (two.float().mean(0) > 0.95) & (W["tilt"].mean(0) < 10)
        out["two-feet OK (last 3 s both down, tilt < 10 deg)"] = f"{int(ok.sum())}/{len(alive)}"
        out["tilt deg p50"] = round(float(W["tilt"].median()), 1)
    out["mean |delta action| per step (rad)"] = round(float(W["dact"].mean()), 3)
    out["max |action| p95 (rad)"] = round(float(W["act_abs"].quantile(0.95)), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument("--push", type=float, default=0.0, help="velocity kick +-m/s every 3-6 s (0 = none)")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    for sc in a.scenario:
        R, alive, side, dt = run(a.ckpt, sc, N=a.num_envs, T=a.seconds, push=a.push, task=a.task)
        print(f"== {a.ckpt.split('/')[-1]} | {sc} | push {a.push}")
        for k, v in summarize(R, alive, side, dt, sc).items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
