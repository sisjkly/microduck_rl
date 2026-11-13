"""Rehearse a flamingo policy set through the robot's OWN runtime (duck_runtime) in simulation.

    uv run python scripts/flamingo/rehearse_runtime.py policies_flamingo
    uv run python scripts/flamingo/rehearse_runtime.py policies_official_flamingo --seeds 8 --supply robot
    uv run python scripts/flamingo/rehearse_runtime.py policies_flamingo --servo xl330 --profile official

Same Runner / Controller / policy_set code as `duck_runtime.run` on the robot (manifest parsing,
skill timing, unwind, action scale, low-pass, joint clamp); only the body is SimIO
(MuJoCo + BAM servo + mass profile, all-collision scene, byte-level simulated IMU).
SimIO has no backlash and stiffer contacts than the training env, so it is the conservative
check before the robot; scripts/flamingo/compare_ckpts.py (training env) is the policy-quality one.

Per seed: torque on, 2.5 s ramp to HOME, start the walk slot (on_fall=keep: a robot that
slumped during the open-loop hold is got up by the policy), ~2 s settle, trigger the skill,
1 s, optional push, hold to 6 s, then 3.5 s (unwind 1.5 s + back on the walk slot).
OK = never past 50 deg during the hold, >= 95 % of the last 3 s on one foot (swing foot
>= 20 mm above the support foot), and upright (< 10 deg) at the end.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from duck_runtime.controller import Command, Controller, Tuning  # noqa: E402
from duck_runtime.policy_set import apply_walk_tuning, load_set  # noqa: E402
from duck_runtime.runner import Runner  # noqa: E402
from duck_runtime.sim_io import SUPPLY_PRESETS, SimIO  # noqa: E402

SKILLS = ("flamingo_right", "flamingo_left")     # right = RIGHT foot down


def tilt_deg(g) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))


def episode(pset, name, servo, profile, supply, seed, push):
    io = SimIO(mass_profile=None if profile == "official" else profile, servo=servo, seed=seed,
               **SUPPLY_PRESETS[supply])
    io.reset(joint_noise=0.03)
    r = Runner(io, Controller(pset.walk, apply_walk_tuning(pset, Tuning.official(), None)),
               policies=pset, log=[], on_fall="keep")
    site_id = {n: mujoco.mj_name2id(io.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")}

    def drive(sec, rec=None, tilts=None):
        for _ in range(int(round(sec / io.control_dt))):
            r.set_command(Command(twist=(0.0, 0.0, 0.0)), io.now())
            r.tick()
            if rec is not None:
                rec.append((io.data.site_xpos[site_id["left_foot"]][2], io.data.site_xpos[site_id["right_foot"]][2]))
                tilts.append(tilt_deg(r.log[-1].gravity))

    r.enable()
    drive(2.5)
    if not r.start():
        return {"error": r.refusal}
    drive(2.0 + 0.1 * seed)
    why = r.trigger(name)
    if why:
        return {"error": why}
    rec, tilts = [], []
    drive(1.0, rec, tilts)
    if push:
        io.push(push, 0.0)
    drive(5.0, rec, tilts)
    hold = np.array(rec[-int(round(3.0 / io.control_dt)):])
    drive(3.5)
    right_down = name == "flamingo_right"
    swing, sup = (hold[:, 0], hold[:, 1]) if right_down else (hold[:, 1], hold[:, 0])
    one = float(((swing - sup) > 0.02).mean())
    end = tilt_deg(r.log[-1].gravity)
    return {"ok": max(tilts) < 50 and one > 0.95 and end < 10, "alive": max(tilts) < 50,
            "one_foot": one, "lift": float(np.median(swing - sup)), "end_tilt": end}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="policy set dir (manifest.json with flamingo_right / flamingo_left)")
    ap.add_argument("--servo", choices=["hd1910", "xl330"], default="hd1910")
    ap.add_argument("--profile", default="ours_835g", help="mass profile: ours_835g | official | ...")
    ap.add_argument("--supply", choices=sorted(SUPPLY_PRESETS), default="ideal",
                    help="battery model: ideal (training-like) / robot (this robot's measured sag)")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--push", type=float, default=0.0, help="trunk velocity kick (m/s, x) 1 s into the skill")
    ap.add_argument("--skill", nargs="+", choices=SKILLS, default=list(SKILLS))
    a = ap.parse_args()

    pset = load_set(a.set)
    for name in a.skill:
        rows = [episode(pset, name, a.servo, a.profile, a.supply, s, a.push) for s in range(a.seeds)]
        bad = [x["error"] for x in rows if "error" in x]
        if bad:
            print(f"{name}: refused: {bad[0]}")
            continue
        print(f"{a.servo}/{a.profile}/{a.supply} push {a.push} {name}: ok {sum(x['ok'] for x in rows)}/{a.seeds}"
              f"  alive {sum(x['alive'] for x in rows)}  one_foot {np.mean([x['one_foot'] for x in rows]):.2f}"
              f"  lift {1000 * np.median([x['lift'] for x in rows]):.0f} mm"
              f"  end_tilt {np.median([x['end_tilt'] for x in rows]):.1f}", flush=True)


if __name__ == "__main__":
    main()
