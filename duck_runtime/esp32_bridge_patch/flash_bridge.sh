#!/usr/bin/env bash
# Flash the ESP32-S3 bridge from the Pi, over the same USB cable the runtime uses (works over SSH).
#
#   bash duck_runtime/esp32_bridge_patch/flash_bridge.sh bridge_mouth15.bin            # 15 servos (mouth)
#   bash duck_runtime/esp32_bridge_patch/flash_bridge.sh bridge_stock14.bin            # back to 14
#
# The image is a merged one (bootloader + partitions + app, written at 0x0), built from the
# MDK2 sketch with Arduino-ESP32 3.0.7, board ESP32S3 Dev Module, USB CDC On Boot = Enabled,
# USB Mode = Hardware CDC and JTAG, 4 MB DIO. Through the S3's USB-Serial/JTAG port esptool
# resets the chip into download mode by itself; nothing to press. If it ever cannot connect:
# hold BOOT, tap RESET, release BOOT, run again (or flash from a PC with the Arduino IDE).
set -euo pipefail
BIN=${1:?usage: flash_bridge.sh image.bin [port]}
PORT=${2:-/dev/ttyACM0}
PY=${PY:-$HOME/microduck/.venv/bin/python}
[ -f "$BIN" ] || { echo "no such file: $BIN"; exit 1; }

if pgrep -f "[d]uck_runtime\.(run|dashboard|tools)" >/dev/null; then
  echo "a duck_runtime process still holds the bridge: stop it (Ctrl+C) first"; exit 1
fi
sudo systemctl stop microduck-gamepad.service 2>/dev/null || true

if ! "$PY" -m esptool version >/dev/null 2>&1; then
  echo "installing esptool into $(dirname "$(dirname "$PY")")"
  "$PY" -m pip install -q "esptool==4.8.1"
fi

"$PY" -m esptool --chip esp32s3 --port "$PORT" --baud 921600 \
  --before default_reset --after hard_reset write_flash 0x0 "$BIN"

echo "waiting for the bridge to come back on $PORT..."
for _ in $(seq 1 40); do [ -e "$PORT" ] && break; sleep 0.25; done
sleep 1
cd "$(dirname "$0")/../.."
"$PY" - "$PORT" <<'EOF'
import sys
from duck_runtime.bridge import Bridge
b = Bridge(sys.argv[1])
try:
    st = b.status()
    print(f"bridge up: MDK2 v{st['version']}, serves IDs {b.served_ids()}, fault {st['fault']} ({st['fault_name']})")
finally:
    b.close()
EOF
