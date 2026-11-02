"""Pi board monitor: sysfs temperature / clock and vcgencmd throttle flags, faked."""

import os
import stat

from duck_runtime.board import BoardMonitor, decode_throttled


def fake_vcgencmd(tmp_path, value="0x50005"):
    p = tmp_path / "vcgencmd"
    p.write_text(f"#!/bin/sh\necho throttled={value}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_decode():
    assert decode_throttled(None) == ([], [])
    assert decode_throttled(0) == ([], [])
    now, ever = decode_throttled(0x50005)
    assert now == ["欠压", "降频"] and ever == ["欠压", "降频"]
    assert decode_throttled(0x80000) == ([], ["温度软限"])


def test_poll_reads_sysfs_and_vcgencmd(tmp_path):
    (tmp_path / "temp").write_text("48312\n")
    (tmp_path / "freq").write_text("1000000\n")
    b = BoardMonitor(thermal=tmp_path / "temp", cpufreq=tmp_path / "freq", vcgencmd=fake_vcgencmd(tmp_path))
    b.poll()
    st = b.status()
    assert st == {"temp_c": 48.3, "freq_mhz": 1000, "throttled": "0x50005",
                  "now": ["欠压", "降频"], "ever": ["欠压", "降频"]}


def test_missing_everything_reads_none(tmp_path):
    b = BoardMonitor(thermal=tmp_path / "nope", cpufreq=tmp_path / "nope2", vcgencmd="no-such-vcgencmd-here")
    b.poll()
    assert b.status() == {"temp_c": None, "freq_mhz": None, "throttled": None, "now": [], "ever": []}


def test_thread_runs_and_stops(tmp_path):
    (tmp_path / "temp").write_text("51000")
    b = BoardMonitor(period_s=0.01, thermal=tmp_path / "temp", cpufreq=tmp_path / "x", vcgencmd=None).start()
    import time
    for _ in range(100):
        if b.temp_c is not None:
            break
        time.sleep(0.01)
    b.stop()
    assert b.temp_c == 51.0
