#!/usr/bin/env python3
"""Add the mouth servo (a 15th HD1910) to the MDK2 ESP32-S3 bridge sketch.

The bridge (microduck_esp32_bridge, MicroDuck_XGO_r1) serves exactly 14 servos, IDs 1-14
(bridge_config.h: SERVO_COUNT = 14, "no mouth joint in its 14-action policy"), so a mouth
servo on the same bus is never polled or driven. This edits bridge_config.h only:

  SERVO_COUNT 14 -> 15, DXL_IDS / BUS_IDS += mouth ID, JOINT_DIR += -1,
  FT_AT_DXL_CENTER += 2048, FT_MIN += 0, FT_MAX += 4095, CALIBRATED_MASK -> 0x7FFF,
  static_assert(SERVO_COUNT <= 14) -> <= 16 (every mask in bridge_runtime.cpp is uint16_t;
  packet buffers are sized by SERVO_COUNT).

Your existing values for servos 1-14 are kept byte for byte. The mouth's center/direction
are placeholders: duck_runtime reads the firmware map back and calibrates in Python
(``duck_runtime.tools zero --joints mouth`` / ``direction --joints mouth``).

Usage, on the computer with the sketch you flashed before:
    python3 add_mouth.py /path/to/microduck_esp32_bridge            # mouth = ID 15
    python3 add_mouth.py /path/to/microduck_esp32_bridge --id 15
Backs up bridge_config.h to bridge_config.h.before_mouth. Re-running is a no-op.
Then flash as before: ESP32S3 Dev Module, USB CDC On Boot = Enabled, native USB port.

After flashing, the host must name all 15 servos when it turns torque off (that is what
clears a latched fault): duck_runtime does this by pinging which IDs the bridge serves.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ARRAYS = {  # name -> value appended for the mouth (None = the mouth ID)
    "DXL_IDS": None, "BUS_IDS": None, "JOINT_DIR": "-1",
    "FT_AT_DXL_CENTER": "2048", "FT_MIN": "0", "FT_MAX": "4095",
}


def patch(text: str, mouth_id: int) -> str:
    m = re.search(r"constexpr\s+int\s+SERVO_COUNT\s*=\s*(\d+)\s*;", text)
    if not m:
        raise SystemExit("bridge_config.h: SERVO_COUNT not found — is this the MDK2 bridge sketch?")
    count = int(m.group(1))
    if count == 15:
        ids = re.search(r"BUS_IDS\[SERVO_COUNT\]\s*=\s*\{([^}]*)\}", text)
        if ids and str(mouth_id) in [v.strip() for v in ids.group(1).split(",")]:
            print("already patched (SERVO_COUNT = 15 with the mouth ID): nothing to do")
            return text
        raise SystemExit("SERVO_COUNT is already 15 but without this ID — check bridge_config.h by hand")
    if count != 14:
        raise SystemExit(f"SERVO_COUNT = {count}, expected 14")
    line_end = text.find("\n", m.start())
    line_end = len(text) if line_end < 0 else line_end
    text = (text[:m.start()] + f"constexpr int SERVO_COUNT = 15;     // 14 joints + mouth (ID {mouth_id}), add_mouth.py"
            + text[line_end:])
    for name, value in ARRAYS.items():
        pat = re.compile(r"(constexpr\s+\w+\s+" + name + r"\[SERVO_COUNT\]\s*=\s*\{)([^}]*)(\})", re.S)
        a = pat.search(text)
        if not a:
            raise SystemExit(f"bridge_config.h: array {name}[SERVO_COUNT] not found")
        body = a.group(2)
        vals = [v.strip() for v in body.split(",") if v.strip()]
        if len(vals) != 14:
            raise SystemExit(f"{name} has {len(vals)} entries, expected 14")
        if name in ("DXL_IDS", "BUS_IDS") and str(mouth_id) in vals:
            raise SystemExit(f"ID {mouth_id} is already used in {name}")
        add = str(mouth_id) if value is None else value
        stripped = body.rstrip()
        trailing = body[len(stripped):]
        new_body = stripped + ("," if not stripped.endswith(",") else "") + add + trailing
        text = text[:a.start(2)] + new_body + text[a.end(2):]
    text, n = re.subn(r"(constexpr\s+uint16_t\s+CALIBRATED_MASK\s*=\s*)0x[0-9A-Fa-f]+", r"\g<1>0x7FFF", text)
    if n != 1:
        raise SystemExit("CALIBRATED_MASK not found")
    text, n = re.subn(r"static_assert\(SERVO_COUNT\s*<=\s*14", "static_assert(SERVO_COUNT <= 16", text)
    if n != 1:
        print("note: static_assert(SERVO_COUNT <= 14) not found; check that the sketch allows 15", file=sys.stderr)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sketch", help="folder with microduck_esp32_bridge.ino and bridge_config.h")
    ap.add_argument("--id", type=int, default=15, help="mouth servo bus ID (default 15)")
    args = ap.parse_args(argv)
    if not 1 <= args.id <= 253:
        raise SystemExit("ID must be 1..253")
    cfg = Path(args.sketch) / "bridge_config.h"
    if not cfg.is_file():
        raise SystemExit(f"{cfg} not found")
    old = cfg.read_text()
    new = patch(old, args.id)
    if new == old:
        return
    backup = cfg.with_name("bridge_config.h.before_mouth")
    if not backup.exists():
        shutil.copy2(cfg, backup)
    cfg.write_text(new)
    print(f"patched {cfg} (backup: {backup.name}); now flash the sketch")


if __name__ == "__main__":
    main()
