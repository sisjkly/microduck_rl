"""Copy the official policy set and add the flamingo skills to the COPY.

Run on the duck, from ~/microduck_rl:
    python3 policies_flamingo/merge_into_official.py     # copy this file into the set dir first
    # options: --official policies --flamingo policies_flamingo --out policies_official_flamingo

- The official set (--official) is only read, never modified.
- The copy keeps everything the official set has (manifest, every .onnx, overrides.json);
  the walk slot stays the official velstand.
- Only the flamingo SKILL entries are appended (flamingo_right, flamingo_left); the walk
  slot from policies_flamingo/manifest.json is NOT copied.
- Refuses to overwrite an existing --out unless --force.
"""

import argparse
import json
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default="policies")
    ap.add_argument("--flamingo", default="policies_flamingo")
    ap.add_argument("--out", default="policies_official_flamingo")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    a = ap.parse_args()

    off_man = os.path.join(a.official, "manifest.json")
    fl_man = os.path.join(a.flamingo, "manifest.json")
    for p in (off_man, fl_man, os.path.join(a.flamingo, "flamingo.onnx")):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 1
    if os.path.exists(a.out):
        if not a.force:
            print(f"{a.out} already exists (use --force to replace it)", file=sys.stderr)
            return 1
        shutil.rmtree(a.out)

    shutil.copytree(a.official, a.out)                     # 1) the copy of the official set
    shutil.copy2(os.path.join(a.flamingo, "flamingo.onnx"), os.path.join(a.out, "flamingo.onnx"))

    with open(off_man) as f:
        man = json.load(f)
    with open(fl_man) as f:
        fl = json.load(f)
    if (man.get("schema_version"), man.get("obs_len"), man.get("action_len")) != (2, 61, 14):
        print(f"{off_man}: expected schema 2, 61 -> 14", file=sys.stderr)
        return 1

    existing = {e.get("name") for e in man.get("policies", [])}
    added = []
    for e in fl["policies"]:
        if e.get("kind") == "perpetual":                    # flamingo's own walk slot: not wanted
            continue
        if e["name"] in existing:
            print(f"official set already has a policy named {e['name']!r}", file=sys.stderr)
            return 1
        man["policies"].append(e)                           # 2) add the flamingo skills
        added.append(e["name"])
    man["description"] = (man.get("description", "") + " + flamingo (ours, model_1000)").strip()

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"{a.out}: official set + {', '.join(added)}")

    try:                                                    # 3) load it exactly as the runtime will
        sys.path.insert(0, os.getcwd())
        from duck_runtime.policy_set import bind_keys, load_set
        p = load_set(a.out)
        print("walk:", p.walk_file, "| sitstand:", "yes" if p.sitstand else "no")
        for k, n in bind_keys(p.skills).items():
            s = p.skills[n]
            print(f"  key {k} = {n} ({s.duration_s:g} s)")
    except ImportError:
        print("(duck_runtime not importable from here: run from ~/microduck_rl to check the key list)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
