"""The duck's camera for the dashboard: live MJPEG, photos, recordings. No Python camera stack.

One ``rpicam-vid --codec mjpeg -o -`` process feeds everything (live view and recording share
it, so recording never interrupts the live view). It runs only while the live view is on or a
recording is going, at low priority (nice 10), so the 50 Hz control loop keeps the CPU.

  - live:      the latest JPEG frame, fanned out to every browser (multipart/x-mixed-replace)
  - photo:     ``rpicam-still`` at full resolution; the stream pauses ~2 s if it was on.
               During a recording the current stream frame is saved instead (never interrupts)
  - recording: the stream's JPEG frames written into a QuickTime .mov (Motion-JPEG, one frame
               per chunk, real per-frame timestamps) by ``MovWriter``; plays in QuickTime and
               VLC without re-encoding, and needs no ffmpeg (the duck has none; rpicam-apps is
               built without libav)

This build (2026-10-30): IMX219 160 deg without an IR-cut filter -> the stock tuning is pink;
``imx219_noir.json`` gives natural colour, so it is the default when present.
"""

from __future__ import annotations

import os
import shlex
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

NOIR_TUNING = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
SOI, EOI = b"\xff\xd8", b"\xff\xd9"
MAX_RECORDING_BYTES = 2_000_000_000          # 32-bit chunk offsets; ~45 min at 640x480/15 fps
MAX_BUFFER = 4_000_000                       # drop garbage that never forms a frame


def _jpeg_size(frame: bytes) -> tuple[int, int] | None:
    """Width, height from the SOF marker of a baseline/progressive JPEG."""
    i = 2
    while i + 9 < len(frame):
        if frame[i] != 0xFF:
            return None
        marker = frame[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", frame[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = struct.unpack(">H", frame[i + 2:i + 4])[0]
        i += 2 + seg
    return None


class MovWriter:
    """Motion-JPEG in a QuickTime .mov, written as frames arrive: ftyp, mdat (frames appended,
    size patched at close), moov at the end with per-frame durations (stts) from the frames'
    own timestamps, so dropped frames do not speed the video up."""

    TIMESCALE = 1000                          # ms

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f = open(self.path, "wb")
        self.f.write(self._atom(b"ftyp", b"qt  " + struct.pack(">I", 0) + b"qt  "))
        self.mdat_at = self.f.tell()
        self.f.write(struct.pack(">I", 0) + b"mdat")           # size patched in close()
        self.sizes: list[int] = []
        self.offsets: list[int] = []
        self.stamps: list[float] = []
        self.width = self.height = 0
        self.bytes = 0
        self.closed = False

    @staticmethod
    def _atom(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + kind + payload

    def add(self, frame: bytes, t: float) -> None:
        if self.closed:
            return
        if not self.width:
            wh = _jpeg_size(frame)
            if wh:
                self.width, self.height = wh
        self.offsets.append(self.f.tell())
        self.f.write(frame)
        self.sizes.append(len(frame))
        self.stamps.append(t)
        self.bytes += len(frame)

    @property
    def frames(self) -> int:
        return len(self.sizes)

    @property
    def seconds(self) -> float:
        return (self.stamps[-1] - self.stamps[0]) if len(self.stamps) > 1 else 0.0

    def _durations(self) -> list[int]:
        ts = self.stamps
        if len(ts) < 2:
            return [self.TIMESCALE // 15] * len(ts)
        d = [max(1, round((b - a) * self.TIMESCALE)) for a, b in zip(ts, ts[1:])]
        d.append(round(sum(d) / len(d)))
        return d

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        end = self.f.tell()
        self.f.seek(self.mdat_at)
        self.f.write(struct.pack(">I", end - self.mdat_at))
        self.f.seek(end)
        self.f.write(self._moov())
        self.f.close()

    def _moov(self) -> bytes:
        A = self._atom
        durations = self._durations()
        total = sum(durations)
        now = int(time.time()) + 2082844800                    # seconds since 1904
        matrix = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
        # version/flags, created, modified, timescale, duration, rate 1.0, volume 1.0, reserved,
        # matrix, preview/poster/selection/current times (6 x 0), next track id
        mvhd = A(b"mvhd", struct.pack(">IIIIIIH", 0, now, now, self.TIMESCALE, total, 0x10000, 0x100)
                 + b"\0" * 10 + matrix + b"\0" * 24 + struct.pack(">I", 2))
        tkhd = A(b"tkhd", struct.pack(">IIIIII", 3, now, now, 1, 0, total) + b"\0" * 8
                 + struct.pack(">HHHH", 0, 0, 0, 0) + matrix
                 + struct.pack(">II", self.width << 16, self.height << 16))
        mdhd = A(b"mdhd", struct.pack(">IIIIIHH", 0, now, now, self.TIMESCALE, total, 0, 0))
        hdlr = A(b"hdlr", struct.pack(">I", 0) + b"mhlr" + b"vide" + b"\0" * 12 + b"\x0cVideoHandler")
        vmhd = A(b"vmhd", struct.pack(">IHHHH", 1, 0x40, 0x8000, 0x8000, 0x8000))
        dhlr = A(b"hdlr", struct.pack(">I", 0) + b"dhlr" + b"alis" + b"\0" * 12 + b"\x0bDataHandler")
        dinf = A(b"dinf", A(b"dref", struct.pack(">II", 0, 1) + A(b"alis", struct.pack(">I", 1))))
        name = b"Photo - JPEG"
        entry = (b"\0" * 6 + struct.pack(">H", 1) + struct.pack(">HH", 0, 0) + b"appl"
                 + struct.pack(">II", 0, 512) + struct.pack(">HH", self.width, self.height)
                 + struct.pack(">II", 72 << 16, 72 << 16) + struct.pack(">IH", 0, 1)
                 + bytes([len(name)]) + name + b"\0" * (31 - len(name)) + struct.pack(">hh", 24, -1))
        stsd = A(b"stsd", struct.pack(">II", 0, 1) + A(b"jpeg", entry))
        runs: list[list[int]] = []
        for d in durations:
            if runs and runs[-1][1] == d:
                runs[-1][0] += 1
            else:
                runs.append([1, d])
        stts = A(b"stts", struct.pack(">II", 0, len(runs)) + b"".join(struct.pack(">II", n, d) for n, d in runs))
        stsc = A(b"stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        stsz = A(b"stsz", struct.pack(">III", 0, 0, len(self.sizes)) + b"".join(struct.pack(">I", s) for s in self.sizes))
        stco = A(b"stco", struct.pack(">II", 0, len(self.offsets)) + b"".join(struct.pack(">I", o) for o in self.offsets))
        stbl = A(b"stbl", stsd + stts + stsc + stsz + stco)
        minf = A(b"minf", vmhd + dhlr + dinf + stbl)
        mdia = A(b"mdia", mdhd + hdlr + minf)
        trak = A(b"trak", tkhd + mdia)
        return A(b"moov", mvhd + trak)


class Camera:
    """Owns the camera processes. Thread-safe; every method returns quickly except photo()."""

    def __init__(self, media_dir: str | Path, size=(640, 480), fps: int = 15, tuning: str | None = "auto",
                 extra_args: str = "", still_size=(1640, 1232), vid_cmd: str = "rpicam-vid",
                 still_cmd: str = "rpicam-still", nice: int = 10):
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.size, self.fps, self.still_size = tuple(size), int(fps), tuple(still_size)
        if tuning == "auto":
            tuning = NOIR_TUNING if os.path.exists(NOIR_TUNING) else None
        self.tuning = tuning
        self.extra = shlex.split(extra_args) if extra_args else []
        self.vid_cmd, self.still_cmd = shlex.split(vid_cmd), shlex.split(still_cmd)
        self.nice = nice
        self.lock = threading.RLock()
        self.cond = threading.Condition()
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None
        self.frame: bytes | None = None
        self.frame_t = 0.0
        self.frame_seq = 0
        self.live = False                 # the user's toggle
        self.writer: MovWriter | None = None
        self.rec_name: str | None = None
        self.error = ""
        self._fps_times: list[float] = []
        self._stderr_tail: list[str] = []

    @staticmethod
    def available(vid_cmd: str = "rpicam-vid") -> bool:
        return shutil.which(shlex.split(vid_cmd)[0]) is not None

    # -- process -------------------------------------------------------------------------
    def _tuning_args(self) -> list[str]:
        return ["--tuning-file", self.tuning] if self.tuning else []

    def _start_proc(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        w, h = self.size
        cmd = self.vid_cmd + ["-n", "-t", "0", "--codec", "mjpeg", "--width", str(w), "--height", str(h),
                              "--framerate", str(self.fps)] + self._tuning_args() + self.extra + ["-o", "-"]
        nice = self.nice
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                     preexec_fn=(lambda: os.nice(nice)) if nice else None)
        self.error = ""
        self._stderr_tail = []
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def _stop_proc(self) -> None:
        p = self.proc
        self.proc = None
        if p is None:
            return
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(2)
        if self.reader is not None:
            self.reader.join(3)
        with self.cond:
            self.frame = None
            self.cond.notify_all()

    def _drain_stderr(self, p: subprocess.Popen) -> None:
        for line in iter(p.stderr.readline, b""):
            s = line.decode(errors="replace").strip()
            if s:
                self._stderr_tail = (self._stderr_tail + [s])[-8:]

    def _read(self, p: subprocess.Popen) -> None:
        buf = b""
        while True:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.find(SOI)
                if s < 0:
                    buf = buf[-1:]
                    break
                e = buf.find(EOI, s + 2)
                if e < 0:
                    buf = buf[s:] if len(buf) - s < MAX_BUFFER else b""
                    break
                self._on_frame(buf[s:e + 2])
                buf = buf[e + 2:]
        if self.proc is p:                     # ended by itself, not by stop: say why
            tail = " | ".join(self._stderr_tail[-3:])
            self.error = f"camera process exited ({p.poll()}){': ' + tail if tail else ''}"
            with self.lock:
                self._finish_recording()
                self.live = False
                self.proc = None
            with self.cond:
                self.frame = None
                self.cond.notify_all()

    def _on_frame(self, frame: bytes) -> None:
        t = time.monotonic()
        with self.cond:
            self.frame, self.frame_t = frame, t
            self.frame_seq += 1
            self.cond.notify_all()
        self._fps_times = (self._fps_times + [t])[-30:]
        w = self.writer
        if w is not None:
            w.add(frame, t)
            if w.bytes >= MAX_RECORDING_BYTES:
                with self.lock:
                    self._finish_recording()
                    self.error = "recording stopped at the 2 GB limit"

    def _keep_running(self) -> None:
        if self.live or self.writer is not None:
            self._start_proc()
        else:
            self._stop_proc()

    # -- user actions ----------------------------------------------------------------------
    def set_live(self, on: bool) -> dict:
        with self.lock:
            self.live = bool(on)
            self._keep_running()
        return self.status()

    def wait_frame(self, after_seq: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.frame_seq != after_seq or self.proc is None, timeout=timeout)
            return self.frame_seq, self.frame

    def latest(self, timeout: float = 3.0) -> bytes | None:
        with self.cond:
            if self.frame is None and self.proc is not None:
                self.cond.wait_for(lambda: self.frame is not None or self.proc is None, timeout=timeout)
            return self.frame

    def _name(self, ext: str) -> str:
        return time.strftime(f"duck_%Y%m%d-%H%M%S.{ext}")

    def photo(self) -> dict:
        """Full-resolution still; during a recording, the current stream frame instead."""
        with self.lock:
            if self.writer is not None:
                frame = self.latest()
                if frame is None:
                    return {"ok": False, "error": "no frame yet"}
                name = self._name("jpg")
                (self.media_dir / name).write_bytes(frame)
                return {"ok": True, "file": name, "size": list(self.size), "from": "stream (recording)"}
            was_running = self.proc is not None
            self._stop_proc()
            name = self._name("jpg")
            w, h = self.still_size
            cmd = self.still_cmd + ["-n", "-t", "1000", "--width", str(w), "--height", str(h)] \
                + self._tuning_args() + self.extra + ["-o", str(self.media_dir / name)]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=20)
                ok = r.returncode == 0 and (self.media_dir / name).exists()
                err = "" if ok else (r.stderr.decode(errors="replace").strip().splitlines() or ["failed"])[-1]
            except (OSError, subprocess.TimeoutExpired) as e:
                ok, err = False, str(e)
            if was_running:
                self._keep_running()
            if not ok:
                self.error = f"photo failed: {err}"
                return {"ok": False, "error": self.error}
            return {"ok": True, "file": name, "size": [w, h], "from": "still"}

    def start_recording(self) -> dict:
        with self.lock:
            if self.writer is not None:
                return {"ok": True, "file": self.rec_name}
            self.rec_name = self._name("mov")
            self.writer = MovWriter(self.media_dir / self.rec_name)
            self._keep_running()
            return {"ok": True, "file": self.rec_name}

    def _finish_recording(self) -> dict | None:
        w, name = self.writer, self.rec_name
        if w is None:
            return None
        self.writer = None
        w.close()
        if w.frames == 0:
            w.path.unlink(missing_ok=True)
            return {"file": None, "frames": 0, "seconds": 0.0}
        return {"file": name, "frames": w.frames, "seconds": round(w.seconds, 1)}

    def stop_recording(self) -> dict:
        with self.lock:
            done = self._finish_recording()
            self._keep_running()
        if done is None:
            return {"ok": True, "file": None}
        return {"ok": True, **done}

    def close(self) -> None:
        with self.lock:
            self._finish_recording()
            self.live = False
            self._stop_proc()

    # -- reporting -------------------------------------------------------------------------
    def status(self) -> dict:
        ts = self._fps_times
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else None
        stale = self.frame_t and time.monotonic() - self.frame_t > 2.0
        w = self.writer
        return {
            "running": self.proc is not None, "live": self.live,
            "fps": None if fps is None or stale else round(fps, 1),
            "size": list(self.size), "tuning": os.path.basename(self.tuning) if self.tuning else "default",
            "recording": ({"file": self.rec_name, "seconds": round(w.seconds, 1), "frames": w.frames,
                           "mb": round(w.bytes / 1e6, 1)} if w is not None else None),
            "error": self.error,
        }

    def list_media(self, limit: int = 40) -> list[dict]:
        out = []
        files = [p for p in self.media_dir.glob("duck_*") if p.suffix in (".jpg", ".mov")]
        for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            if self.writer is not None and p.name == self.rec_name:
                continue                      # still being written
            st = p.stat()
            out.append({"name": p.name, "kind": "photo" if p.suffix == ".jpg" else "video",
                        "bytes": st.st_size, "mtime": st.st_mtime})
        return out
