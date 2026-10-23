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

# esptool: one already on PATH, else a small venv of its own (the robot venv is uv-made, no pip).
ESPTOOL=""
for c in esptool esptool.py; do command -v "$c" >/dev/null 2>&1 && { ESPTOOL="$c"; break; }; done
if [ -z "$ESPTOOL" ]; then
  EV="$HOME/.local/share/duck-esptool"
  if [ ! -x "$EV/bin/esptool.py" ] && [ ! -x "$EV/bin/esptool" ]; then
    echo "installing esptool into $EV"
    if python3 -m venv "$EV" 2>/dev/null; then
      "$EV/bin/python" -m pip install -q "esptool==4.8.1"
    elif command -v uv >/dev/null 2>&1; then
      uv venv -q "$EV" && uv pip install -q --python "$EV/bin/python" "esptool==4.8.1"
    else
      echo "cannot make a venv: run  sudo apt install -y python3-venv  (or  sudo apt install -y esptool ) and retry"
      exit 1
    fi
  fi
  ESPTOOL="$EV/bin/esptool.py"; [ -x "$ESPTOOL" ] || ESPTOOL="$EV/bin/esptool"
fi

"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  --before default_reset --after hard_reset write_flash 0x0 "$BIN"

echo "waiting for the bridge to boot on $PORT..."
sleep 2
cd "$(dirname "$0")/../.."
"$PY" - "$PORT" <<'EOF'
import sys, time
from duck_runtime.bridge import Bridge, BridgeError
last = None
for attempt in range(20):                      # USB re-enumeration + boot can take a few seconds
    try:
        b = Bridge(sys.argv[1])
        try:
            st = b.status()
            print(f"bridge up: MDK2 v{st['version']}, serves IDs {b.served_ids()}, fault {st['fault']} ({st['fault_name']})")
            sys.exit(0)
        finally:
            b.close()
    except (BridgeError, OSError) as e:
        last = e
        time.sleep(0.5)
print(f"bridge not answering after 10 s ({last}). Unplug/replug the ESP32 USB (or power-cycle the robot), "
      "then: duck duck_runtime.tools scan")
sys.exit(1)
EOF
