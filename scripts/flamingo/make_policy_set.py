"""Package an exported flamingo ONNX as a robot-side policy set (manifest schema 2).

    uv run python scripts/flamingo/make_policy_set.py flamingo.onnx policies_flamingo --servos xl330

Writes <out>/flamingo.onnx + <out>/manifest.json (+ a copy of merge_into_official.py):
  - walk slot      = flamingo itself at twist [0,0,0] (two-feet stand; a standalone set)
  - flamingo_right = episodic skill, twist [1, +1, 0]: RIGHT foot down, left leg lifts back
  - flamingo_left  = episodic skill, twist [1, -1, 0]: LEFT foot down, right leg lifts back
  each held `--duration` s, then `--unwind` s at [0,0,0] (both feet down) before handing back.
action_scale 1.0 and no low-pass: what the policy was trained with.
In this standalone set the walk keys would drive the flamingo command (vx → flag), so do not
use them; merge_into_official.py adds only the two skills to a copy of the official set.
`--servos` is a label (shown by the runtime), not a switch: give the servo model the
checkpoint was TRAINED with (`flamingo` branch → xl330, `flamingo-hd1910` → hd1910).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

TASK_ID = "Mjlab-Flamingo-Flat-Backlash-MicroDuck"
UNFILTERED = {"head": 1.0, "legs": 1.0}


def skill(name: str, side: float, duration: float, unwind: float) -> dict:
    return {"file": "flamingo.onnx", "name": name, "kind": "episodic", "duration_s": duration,
            "action_scale": 1.0, "lowpass": UNFILTERED, "unwind_s": unwind,
            "command": {"encoding": "constant", "twist": [1.0, side, 0.0], "idle": [0.0, 0.0, 0.0]},
            "training": {"task_id": TASK_ID}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--servos", choices=["xl330", "hd1910"], required=True)
    ap.add_argument("--duration", type=float, default=6.0, help="one-foot hold per key press (s)")
    ap.add_argument("--unwind", type=float, default=1.5, help="two-feet settle before handing back (s)")
    ap.add_argument("--note", default="", help="free text appended to the description (e.g. run @ checkpoint)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    shutil.copy2(a.onnx, os.path.join(a.out, "flamingo.onnx"))
    man = {
        "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
        "description": f"flamingo (single-leg stand), own set; walk slot = flamingo two-feet stand {a.note}".strip(),
        "robot": {"servos": a.servos, "control_hz": 50},
        "policies": [
            {"file": "flamingo.onnx", "name": "flamingo_stand", "kind": "perpetual", "slot": "walk",
             "entry_pose": "standing", "action_scale": 1.0, "lowpass": UNFILTERED},
            skill("flamingo_right", 1.0, a.duration, a.unwind),
            skill("flamingo_left", -1.0, a.duration, a.unwind),
        ],
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    # ship the merge tool with the set, so the robot needs nothing from this branch
    shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_into_official.py"), a.out)
    print(f"wrote {a.out}/flamingo.onnx + manifest.json ({a.servos}) + merge_into_official.py")


if __name__ == "__main__":
    main()
