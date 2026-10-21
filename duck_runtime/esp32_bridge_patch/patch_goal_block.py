#!/usr/bin/env python3
"""Patch the MDK2 ESP32-S3 bridge so every goal write also zeroes HD1910 44..47.

Why: the bridge writes only the 2-byte position at 42..43, so registers 44..45 (HLS-style
goal current, read as 500 on all 14 servos = 3.25 A cap) and 46..47 (goal speed) keep
whatever is in RAM. kissqy's HD1910 runtime -- the one microduck_rl's HD1910 actuator
model (P=5) was identified against and which walks -- writes the 6-byte block
[position, 0, 0, 0, 0] (duck-control/src/feetech.rs positions(), hd1910.rs
goal_length 6, goal_speed_raw 0). This patch makes the bridge do the same.

Usage (on the computer with the Arduino sketch you flashed):
    python3 patch_goal_block.py /path/to/microduck_esp32_bridge
It backs up bridge_runtime.cpp to bridge_runtime.cpp.before_goal_block and edits only
syncPositions(). Re-running is a no-op. Then flash as before (ESP32S3 Dev Module,
USB CDC On Boot = Enabled, native USB port).
"""
import shutil
import sys
from pathlib import Path

OLD = '''static bool syncPositions(const uint8_t *positions,uint16_t mask) {
  // HD1910: only addresses 42..43 are position. Never zero 44..47 here:
  // these are current/speed controls, not padding for a generic STS move.
  return syncValues(42,2,positions,mask);
}'''

NEW = '''static bool syncPositions(const uint8_t *positions,uint16_t mask) {
  // HD1910 goal block 42..47 = position, goal current 0, goal speed 0, exactly as
  // kissqy's verified HD1910 runtime writes it (microduck_rl's P=5 actuator model was
  // identified this way). Leaving 44..45 untouched kept a 500 (3.25 A) current cap.
  uint8_t rows[SERVO_COUNT*6]={};
  for(int i=0;i<SERVO_COUNT;++i){rows[i*6]=positions[i*2];rows[i*6+1]=positions[i*2+1];}
  return syncValues(42,6,rows,mask);
}'''


def main(sketch: str) -> int:
    path = Path(sketch) / "bridge_runtime.cpp"
    src = path.read_text()
    if NEW in src:
        print(f"{path}: already patched")
        return 0
    if src.count(OLD) != 1:
        print(f"{path}: syncPositions() is not the expected MDK2 version; not touching it", file=sys.stderr)
        return 1
    backup = path.with_name("bridge_runtime.cpp.before_goal_block")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(src.replace(OLD, NEW))
    print(f"patched {path} (backup: {backup.name}); now flash the sketch")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
