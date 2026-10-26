"""Download a third-party policy set and write a manifest our ``--set`` loader reads.

  duck duck_runtime.fetch_set kissqy                      # -> policies_hd1910/
  duck duck_runtime.fetch_set kissqy --tar Microduck-Robotd-FT6-R17.tar.gz   # already downloaded

kissqy (github.com/kissqy/Microduck-kissqy) trained the whole set for HD1910 servos with
microduck_rl's tasks: BAM hd1910_m6, firmware P=5, 820 g, the official 61 -> 14 contract and
the official HOME (checked: identical to contract.DEFAULT_POSITION). The six networks below are
the ones his R17 V167 firmware runs (packaging/robotd-profile.toml), pinned by SHA-256 and
taken from his robotd package (the only place that also holds kick_right).

Deployment settings are his (firmware/docs/design/r17-settings-design.md and each model's
manifest): walk scale 1.0 with head/legs low-pass 0.5/0.7 (deployment-execution.json);
ground pick, kicks and roulade scale 1.0 unfiltered; sit/stand on the stock sit/stand path
(global filter). Personal, non-commercial use only (his LICENSE).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request

KISSQY_TAR_URL = ("https://github.com/kissqy/Microduck-kissqy/raw/main/"
                  "microduck-hd1910/firmware/console/updates/Microduck-Robotd-FT6-R17.tar.gz")

UNFILTERED = {"head": 1.0, "legs": 1.0}

# name -> (sha256, manifest entry). Order = the official listing order of the set.
KISSQY_SET = {
    "velstand": ("1cbf8730eadd75a5704332c9f3c10f60bfa9412135b907831c9dbd731ce1795f", {
        "kind": "perpetual", "slot": "walk", "entry_pose": "standing",
        "action_scale": 1.0, "lowpass": {"head": 0.5, "legs": 0.7},
        "training": {"task_id": "Mjlab-VelStand-Rough-Backlash-MicroDuck", "checkpoint": 9599}}),
    "sitstand": ("1c94934c6421ec72ed48079cea74966868a7a59f7d732e0207d0f7bdaeed3c50", {
        "name": "sitstand", "kind": "scripted",
        "command": {"encoding": "posture_flag", "slot": "twist.vx", "sit": 1.0, "stand": 0.0,
                    "idle": [0.0, 0.0, 0.0]},
        "ramp_s": 2.0, "unwind_s": 1.0, "action_scale": 1.0,
        "training": {"task_id": "Mjlab-SitStand-Flat-MicroDuck", "checkpoint": 10000}}),
    "ground_pick": ("b14e3a8b56ddbb8d2d831e727774a168c8f5b00194e3e431ae7593a840498c69", {
        "name": "ground_pick", "kind": "episodic", "duration_s": 2.8, "action_scale": 1.0,
        "lowpass": UNFILTERED,
        "command": {"encoding": "phase", "slots": "twist.vx,twist.vy", "period_s": 4.0, "end_phase": 0.7},
        "training": {"task_id": "Mjlab-GroundPick-Flat-Backlash-MicroDuck"}}),
    "roulade": ("59e15d7b04d0c40e9b11b8a70bdecae6da87a5e096efc204fb58eea304fcbbac", {
        "name": "roulade", "kind": "episodic", "duration_s": 1.0, "action_scale": 1.0,
        "lowpass": UNFILTERED, "training": {"task_id": "Mjlab-Roulade-Flat-MicroDuck"}}),
    "kick_left": ("ee0376c927044cdece056ba1d3e6eac77b3e02ee3f701f1feb437c183ed39a6d", {
        "name": "kick_left", "kind": "episodic", "duration_s": 0.5, "action_scale": 1.0,
        "lowpass": UNFILTERED, "training": {"task_id": "Mjlab-BallKickLeft-Flat-Backlash-MicroDuck"}}),
    "kick_right": ("67ad09d82f4865d222a7ae06f250cf9f77f08c27e4f019ee3f7e431a764419fd", {
        "name": "kick_right", "kind": "episodic", "duration_s": 0.5, "action_scale": 1.0,
        "lowpass": UNFILTERED, "training": {"task_id": "Mjlab-BallKick-Flat-MicroDuck", "checkpoint": 10000}}),
}


def kissqy_manifest() -> dict:
    policies = []
    for name, (_, entry) in KISSQY_SET.items():
        policies.append({"file": f"kissqy_{name}.onnx", **entry})
    return {
        "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "hd1910", "control_hz": 50},
        "description": "kissqy's HD1910-trained set (github.com/kissqy/Microduck-kissqy, R17 V167): "
                       "P=5, 820 g, official contract and HOME. Personal non-commercial use only.",
        "source": KISSQY_TAR_URL,
        "policies": policies,
    }


def _download(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf, got = io.BytesIO(), 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        return buf.getvalue()


def extract_kissqy(tar_bytes: bytes, out_dir: str) -> list[str]:
    """Pull the six pinned networks out of his robotd package into ``out_dir``; write the manifest."""
    wanted = {sha[:32]: (name, sha) for name, (sha, _) in KISSQY_SET.items()}
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if len(parts) >= 3 and parts[-1] == "policy.onnx" and parts[-3] == "models" and parts[-2] in wanted:
                name, sha = wanted[parts[-2]]
                data = tf.extractfile(m).read()
                if hashlib.sha256(data).hexdigest() != sha:
                    raise SystemExit(f"{m.name}: SHA-256 mismatch (package changed upstream?)")
                found[name] = data
    missing = [n for n in KISSQY_SET if n not in found]
    if missing:
        raise SystemExit(f"not in the package: {missing} (upstream release changed; pins are for R17 V167)")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in KISSQY_SET:
        path = os.path.join(out_dir, f"kissqy_{name}.onnx")
        with open(path, "wb") as f:
            f.write(found[name])
        written.append(path)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(kissqy_manifest(), f, indent=2, ensure_ascii=False)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m duck_runtime.fetch_set", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["kissqy"])
    ap.add_argument("--out", default="policies_hd1910", help="target directory (default policies_hd1910)")
    ap.add_argument("--tar", help="use an already-downloaded Microduck-Robotd-FT6-R17.tar.gz")
    args = ap.parse_args(argv)
    if args.tar:
        with open(args.tar, "rb") as f:
            data = f.read()
    else:
        data = _download(KISSQY_TAR_URL)
    for p in extract_kissqy(data, args.out):
        print(f"  {p}")
    print(f"  {os.path.join(args.out, 'manifest.json')}")
    print(f"run: duck duck_runtime.dashboard --calib calib.json --set {args.out} --on-fall getup")


if __name__ == "__main__":
    main(sys.argv[1:])
