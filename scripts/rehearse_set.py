"""Rehearse the official policy set in simulation, through the exact runtime the robot runs.

Same Runner / Controller / policy_set code as ``duck_runtime.run`` on the robot; only the
body is SimIO (MuJoCo + BAM HD1910 servos + a mass profile, all-collision scene, so sitting
and rolling touch the floor).

  # every move, 8 seeds each, this robot (835 g with the mouth, HD1910)
  uv run scripts/rehearse_set.py --set policies
  # one move, watch it (needs a display)
  uv run scripts/rehearse_set.py --set policies --scenario roulade --seeds 1 --viewer
  # the set1.npz fall: a backward shove 0.8 s into the rise
  uv run scripts/rehearse_set.py --set policies --scenario sitstand --push 0.35 --push-at 0.8
  # compare with the original mass distribution
  uv run scripts/rehearse_set.py --set policies --profile official
  # the robot's own safety rule (limp on fall) vs robotd's default (keep driving)
  uv run scripts/rehearse_set.py --set policies --scenario roulade --on-fall keep

Pitch sign: + = nose down, - = leaning back (as in the robot logs). A run counts as
"upright" if it ends in RUN with the trunk within 10 deg of vertical.

--log DIR writes one .npz per episode with the same fields as the robot's --log
(t, state, mode, obs, targets, positions, gravity, gyro), for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from duck_runtime.controller import Command, Controller, Tuning  # noqa: E402
from duck_runtime.policy_set import apply_walk_tuning, load_set  # noqa: E402
from duck_runtime.runner import Runner, State  # noqa: E402
from duck_runtime.sim_io import SimIO  # noqa: E402


SUPPLIES = {"ideal": dict(vin=7.6, vin_drop_gain=0.1, vin_min=6.0),
            "robot": dict(vin=7.5, vin_drop_gain=1.8, vin_min=3.0),
            "robot-full": dict(vin=8.4, vin_drop_gain=1.8, vin_min=3.0)}


def supply_kwargs(args) -> dict:
    return SUPPLIES[getattr(args, "supply", "ideal")]


def tilt_deg(g) -> float:
    return math.degrees(math.acos(float(np.clip(-g[2], -1.0, 1.0))))


def pitch_deg(g) -> float:
    return math.degrees(math.atan2(-g[0], -g[2]))


def episode(pset, args, scenario: str, seed: int) -> dict:
    io = SimIO(mass_profile=None if args.profile == "official" else args.profile, servo="hd1910", seed=seed,
               **supply_kwargs(args))
    io.reset(joint_noise=args.joint_noise)
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(io.model, io.data)
    try:
        return _episode(io, viewer, pset, args, scenario, seed)
    finally:
        if viewer is not None:
            viewer.close()


def _episode(io, viewer, pset, args, scenario: str, seed: int) -> dict:
    tuning = apply_walk_tuning(pset, Tuning.official(), args.action_scale)   # as duck_runtime.run
    r = Runner(io, Controller(pset.walk, tuning), policies=pset, log=[], on_fall=args.on_fall)
    rng = np.random.default_rng(seed)
    cmd = tuple(float(v) for v in args.cmd.split(","))

    def tick():
        r.set_command(Command(twist=cmd), io.now())
        t0 = time.monotonic()
        r.tick()
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, io.control_dt - (time.monotonic() - t0)))

    def drive(sec):
        for _ in range(int(round(sec / io.control_dt))):
            tick()

    r.enable()
    drive(2.5)                                  # ramp to HOME and hold
    if not r.start():
        return {"error": r.refusal}
    drive(1.0 + rng.uniform(0.0, 0.5))          # a little gait history, seed-dependent

    out: dict = {}
    if scenario == "sitstand":
        if r.sit_toggle() is not None:
            return {"error": "no sit/stand policy in the set"}
        drive(2.5 + rng.uniform(0.0, 1.5))
        out["seat_pitch"] = pitch_deg(r.log[-1].gravity)
        n0 = len(r.log)
        r.sit_toggle()
    else:
        n0 = len(r.log)
        why = r.trigger(scenario)
        if why:
            return {"error": why}
    if args.push:
        drive(args.push_at)
        io.push(args.push, 0.0)
        drive(args.after - args.push_at)
    else:
        drive(args.after)

    seg = r.log[n0:]
    if scenario == "sitstand":
        modes = [x.mode for x in seg]
        k = modes.index("walk") if "walk" in modes else len(seg) - 1
        out["handover_pitch"] = pitch_deg(seg[k].gravity)
    tl = [tilt_deg(x.gravity) for x in seg]
    out["max_tilt"] = max(tl)
    out["end_tilt"] = tl[-1]
    out["end_state"] = r.state.value
    out["upright"] = r.state is State.RUN and tl[-1] < 10.0

    if args.log:
        os.makedirs(args.log, exist_ok=True)
        L = r.log
        np.savez_compressed(
            os.path.join(args.log, f"{scenario}_{args.profile}_s{seed}.npz"),
            t=[x.t for x in L], state=[x.state for x in L], mode=[x.mode for x in L],
            obs=np.array([np.full(61, np.nan) if x.obs is None else x.obs for x in L]),
            targets=np.array([np.full(15, np.nan) if x.targets is None else x.targets for x in L]),
            positions=np.array([x.positions for x in L]),
            gravity=np.array([x.gravity for x in L]), gyro=np.array([x.gyro for x in L]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="policy_set", required=True, help="directory with manifest.json + .onnx")
    ap.add_argument("--policy", help="override the walk slot (e.g. policies/sprint.onnx)")
    ap.add_argument("--sitstand", help="override the sit/stand network")
    ap.add_argument("--scenario", default="all",
                    help="sitstand | kick_left | kick_right | ground_pick | roulade | all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--supply", choices=["ideal", "robot", "robot-full"], default="ideal",
                    help="battery sag model: ideal = training-like (7.6 V, floor 6 V); robot = this robot's "
                         "measured sag (set6.npz: 7.5 V idle, ~4.6 V in a rise, 3-4 V in a kick); "
                         "robot-full = same wiring with a fully charged 8.4 V pack")
    ap.add_argument("--profile", default="ours_835g", help="mass profile: ours_835g | ours_813g | official | ...")
    ap.add_argument("--on-fall", choices=["limp", "keep", "getup"], default="limp",
                    help="limp = this robot's runtime default; keep = robotd's")
    ap.add_argument("--action-scale", type=float, default=None)
    ap.add_argument("--cmd", default="0,0,0", help="walk command vx,vy,wz held throughout (use = for negatives)")
    ap.add_argument("--push", type=float, default=0.0, help="trunk velocity kick, m/s along x (+ shoves it back)")
    ap.add_argument("--push-at", type=float, default=0.8, help="seconds after the move starts")
    ap.add_argument("--after", type=float, default=5.0, help="seconds simulated after the move starts")
    ap.add_argument("--joint-noise", type=float, default=0.03, help="rad, initial pose noise")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo viewer, real time (needs a display)")
    ap.add_argument("--log", help="directory for per-episode .npz logs")
    args = ap.parse_args(argv)

    pset = load_set(args.policy_set, walk=args.policy, sitstand=args.sitstand)
    scenarios = (["sitstand"] if pset.sitstand else []) + list(pset.skills) \
        if args.scenario == "all" else args.scenario.split(",")
    print(f"set {args.policy_set}: walk {pset.walk_file}; profile {args.profile}, HD1910; "
          f"on fall: {args.on_fall}; {args.seeds} seeds" + (f"; push {args.push:+.2f} m/s at {args.push_at} s" if args.push else ""))

    for sc in scenarios:
        rows = []
        for seed in range(args.seeds):
            res = episode(pset, args, sc, seed)
            rows.append(res)
        errs = [r["error"] for r in rows if "error" in r]
        if errs:
            print(f"{sc:12s} refused: {errs[0]}")
            continue
        up = sum(r["upright"] for r in rows)
        line = f"{sc:12s} upright {up}/{len(rows)}   max tilt {np.mean([r['max_tilt'] for r in rows]):5.1f} deg"
        if sc == "sitstand":
            line += (f"   seat pitch {np.mean([r['seat_pitch'] for r in rows]):+5.1f}"
                     f"   handover pitch {np.mean([r['handover_pitch'] for r in rows]):+5.1f} deg")
        bad = [i for i, r in enumerate(rows) if not r["upright"]]
        if bad:
            line += f"   not upright: seeds {bad} (end tilt {[round(rows[i]['end_tilt']) for i in bad]})"
        print(line)


if __name__ == "__main__":
    main()
