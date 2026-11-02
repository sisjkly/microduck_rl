"""The Raspberry Pi itself: SoC temperature, CPU clock, under-voltage / throttling flags.

Read on a background thread (temperature and clock every second from sysfs, the firmware's
throttle flags every 5 s through ``vcgencmd get_throttled``), so the control loop never waits
for it. Anything missing (a PC, the simulator) simply reads as None.

get_throttled bits (Raspberry Pi firmware): 0 under-voltage now, 1 ARM clock capped now,
2 throttled now, 3 soft temperature limit now; 16-19 the same "has happened since boot".
Under-voltage matters here: the Pi and the servos share the battery, and a sagging supply
can slow the CPU and with it the 50 Hz loop.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
CPUFREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
FLAG_NAMES = {0: "欠压", 1: "限频", 2: "降频", 3: "温度软限"}


def decode_throttled(value: int | None) -> tuple[list[str], list[str]]:
    """-> (conditions now, conditions that happened since boot), as short Chinese labels."""
    if value is None:
        return [], []
    now = [n for b, n in FLAG_NAMES.items() if value >> b & 1]
    ever = [n for b, n in FLAG_NAMES.items() if value >> (b + 16) & 1]
    return now, ever


class BoardMonitor:
    def __init__(self, period_s: float = 1.0, throttle_every_s: float = 5.0,
                 thermal: Path = THERMAL, cpufreq: Path = CPUFREQ, vcgencmd: str | None = "vcgencmd"):
        self.period_s, self.throttle_every_s = period_s, throttle_every_s
        self.thermal, self.cpufreq = Path(thermal), Path(cpufreq)
        self.vcgencmd = shutil.which(vcgencmd) if vcgencmd else None
        self.temp_c: float | None = None
        self.freq_mhz: float | None = None
        self.throttled: int | None = None
        self._last_throttle = -1e9
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_int(p: Path) -> int | None:
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            return None

    def _read_throttled(self) -> int | None:
        if not self.vcgencmd:
            return None
        try:
            out = subprocess.run([self.vcgencmd, "get_throttled"], capture_output=True, text=True, timeout=2).stdout
            return int(out.strip().split("=")[1], 16)      # "throttled=0x50000"
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            return None

    def poll(self) -> None:
        """One reading (the thread calls this; tests call it directly)."""
        t = self._read_int(self.thermal)
        self.temp_c = None if t is None else round(t / 1000.0, 1)
        f = self._read_int(self.cpufreq)
        self.freq_mhz = None if f is None else round(f / 1000.0)
        now = time.monotonic()
        if now - self._last_throttle >= self.throttle_every_s:
            self._last_throttle = now
            self.throttled = self._read_throttled()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.period_s)

    def start(self) -> "BoardMonitor":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="board-monitor", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        now, ever = decode_throttled(self.throttled)
        return {"temp_c": self.temp_c, "freq_mhz": self.freq_mhz,
                "throttled": None if self.throttled is None else f"0x{self.throttled:x}",
                "now": now, "ever": ever}
