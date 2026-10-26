"""Web dashboard for the duck: every live reading and every control, served from the robot.

Same Runner / Controller / policy set / safety as ``duck_runtime.run``; the browser replaces
the keyboard. One process owns the hardware: stop ``duck_runtime.run`` first.

  # on the robot (Pi), then open the printed URL from any computer on the same network
  duck duck_runtime.dashboard --calib calib.json --set policies --keep-on-fall
  # two sets, switched from the page while the policy is stopped (first one starts active)
  duck duck_runtime.dashboard --calib calib.json --set policies_hd1910 --set policies --on-fall getup
  # rehearse on a PC with the simulated duck (and this robot's battery sag)
  uv run python -m duck_runtime.dashboard --backend sim --set policies --supply robot

Telemetry: Server-Sent Events at 10 Hz (/api/stream). Controls: JSON POSTs that need the
session token printed at startup (the page keeps it after the first visit). Motion commands
time out like the terminal driver's: no fresh command for 0.5 s -> zero twist (the page sends
one every 200 ms while it is open), so closing the tab stops the robot walking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import secrets
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .contract import DEFAULT_POSITION, JOINT_NAMES, MOUTH_INDEX
from .controller import Command
from .runner import State

STATIC = Path(__file__).resolve().parent / "dashboard_static"
FRAME_HZ = 10.0
COMMAND_TTL_S = 0.5
LIMITS = {"vx": 0.3, "vy": 0.2, "wz": 1.0, "head_deg": 60.0}
MOUTH_RANGE_DEG = (-5.0, 30.0)     # MJCF mouth travel; -5 = closed
LOG_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.npz$")


def _num(x, nd=2):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _arr(a, nd=1, scale=1.0):
    if a is None:
        return None
    return [_num(v * scale, nd) for v in np.asarray(a, dtype=float)]


def pitch_roll(g) -> tuple[float, float]:
    return math.degrees(math.atan2(-g[0], -g[2])), math.degrees(math.atan2(g[1], -g[2]))


class Dashboard:
    """Owns the control loop (call ``run`` on the main thread); HTTP threads only queue work."""

    def __init__(self, runner, io, bank, head, *, backend: str, log_dir: str,
                 token: str | None, realtime: bool, calib_path: str | None = None):
        self.runner, self.io, self.bank = runner, io, bank
        self.head_offset = head
        self.backend = backend
        self.realtime = realtime
        self.calib_path = calib_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.actions: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.cond = threading.Condition()
        self.frame: dict | None = None
        self.frame_seq = 0
        self.events: deque = deque(maxlen=300)
        self.event_seq = 0
        self.cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "head": [0.0, 0.0, 0.0, 0.0]}
        self.cmd_wall = -1e9
        self.recording: list | None = None
        self.rec_started = 0.0
        self.tick_times: deque = deque(maxlen=250)
        self.overruns = 0
        self.clients = 0
        self._prev = {}
        self.started = time.time()

    @property
    def pset(self):
        a = self.bank.active
        return a.pset if a is not None else None

    @property
    def skill_keys(self) -> dict:
        a = self.bank.active
        return a.keys if a is not None else {}

    # -- called from HTTP threads ------------------------------------------------------------
    def submit(self, fn) -> dict:
        """Run ``fn()`` on the control thread at the next tick; wait for its result."""
        done = threading.Event()
        box: dict = {}

        def job():
            try:
                box["result"] = fn()
            except Exception as e:  # report, never kill the loop
                box["error"] = f"{type(e).__name__}: {e}"
            done.set()
        self.actions.put(job)
        if not done.wait(2.0):
            return {"ok": False, "error": "control loop busy"}
        if "error" in box:
            return {"ok": False, "error": box["error"]}
        return box.get("result") or {"ok": True}

    def mouth_servo(self) -> bool:
        cal = getattr(self.io, "cal", None)
        return bool(cal is not None and cal.joints[MOUTH_INDEX].id)

    def set_mouth(self, body: dict) -> dict:
        deg = float(body.get("deg"))
        if not math.isfinite(deg):
            raise ValueError("deg")
        deg = max(MOUTH_RANGE_DEG[0], min(MOUTH_RANGE_DEG[1], deg))
        self.runner.mouth_target = math.radians(deg)   # a float store: safe from this thread
        return {"ok": True, "deg": deg, "servo": self.mouth_servo()}

    def set_command(self, body: dict) -> dict:
        c = self.cmd.copy()
        for k in ("vx", "vy", "wz"):
            if k in body:
                v = float(body[k])
                if not math.isfinite(v):
                    raise ValueError(k)
                c[k] = max(-LIMITS[k], min(LIMITS[k], v))
        if "head" in body:
            h = [float(v) for v in body["head"]]
            if len(h) != 4 or not all(math.isfinite(v) for v in h):
                raise ValueError("head")
            c["head"] = [max(-LIMITS["head_deg"], min(LIMITS["head_deg"], v)) for v in h]
        self.cmd = c
        self.cmd_wall = time.monotonic()
        return {"ok": True, "cmd": c}

    def action(self, body: dict) -> dict:
        r = self.runner
        name = body.get("action")

        def enable():
            try:
                r.enable()
            except RuntimeError as e:          # e.g. a latched bridge fault
                r.limp()
                return {"ok": False, "error": f"enable refused: {e}"}
            return {"ok": True}

        def start():
            if r.start():
                return {"ok": True}
            return {"ok": False, "error": f"start refused: {r.refusal or r.state.value}"}

        def outcome(why, label):
            self.log_event("info" if why is None else "warn", label if why is None else f"{label} refused: {why}")
            return {"ok": why is None, "error": why}

        def zero_twist():
            self.cmd = {**self.cmd, "vx": 0.0, "vy": 0.0, "wz": 0.0}

        jobs = {
            "enable": enable, "start": start,
            "stop": lambda: (r.stop(), {"ok": True})[1],
            "limp": lambda: (r.limp(), {"ok": True})[1],
            "sit": lambda: outcome(r.sit_toggle(), "sit/stand"),
        }
        if name == "skill":
            skill = str(body.get("name", ""))
            return self.submit(lambda: (zero_twist(), outcome(r.trigger(skill), skill))[1])
        if name == "set":
            label = str(body.get("name", ""))

            def switch():
                i = self.bank.find(label)
                if i is None:
                    return {"ok": False, "error": f"no policy set {label!r}"}
                if i == self.bank.index:
                    return {"ok": True}
                zero_twist()
                res = outcome(self.bank.switch(r, i), f"policy set -> {label}")
                if res["ok"]:
                    self.log_event("info", self.bank.active.summary())
                return res
            return self.submit(switch)
        if name not in jobs:
            return {"ok": False, "error": f"unknown action {name!r}"}
        if name == "limp":                      # never queue an emergency stop behind anything
            self.cmd = {**self.cmd, "vx": 0.0, "vy": 0.0, "wz": 0.0}
        return self.submit(jobs[name])

    def settings(self, body: dict) -> dict:
        def apply():
            if "on_fall" in body:
                if body["on_fall"] not in ("limp", "keep", "getup"):
                    raise ValueError("on_fall")
                self.runner.on_fall = body["on_fall"]
            if "getup_timeout_s" in body:
                v = body["getup_timeout_s"]
                self.runner.getup_timeout_s = None if not v else float(v)
            self.log_event("info", f"settings: on fall {self.runner.on_fall}, get-up timeout "
                                   f"{self.runner.getup_timeout_s or 'none'}")
            return {"ok": True}
        return self.submit(apply)

    def record(self, body: dict) -> dict:
        on = bool(body.get("on"))

        def apply():
            if on and self.recording is None:
                self.recording = []
                self.runner.log = self.recording
                self.rec_started = time.time()
                self.log_event("info", "recording started")
                return {"ok": True}
            if not on and self.recording is not None:
                from .run import save_log
                log, self.recording, self.runner.log = self.recording, None, None
                name = time.strftime("dash_%Y%m%d-%H%M%S.npz")
                if log:
                    save_log(str(self.log_dir / name), log)
                    self.log_event("info", f"recording saved: {name} ({len(log)} ticks)")
                return {"ok": True, "file": name if log else None}
            return {"ok": True}
        return self.submit(apply)

    def list_logs(self) -> list[dict]:
        out = []
        for p in sorted(self.log_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            st = p.stat()
            out.append({"name": p.name, "bytes": st.st_size, "mtime": st.st_mtime})
        return out

    def info(self) -> dict:
        r = self.runner
        cal = getattr(self.io, "cal", None)
        skills = []
        if self.pset is not None:
            keys = {v: k for k, v in self.skill_keys.items()}
            skills = [{"name": n, "duration_s": sk.duration_s, "key": keys.get(n)} for n, sk in self.pset.skills.items()]
        joints = []
        for j, n in enumerate(JOINT_NAMES):
            row = {"index": j, "name": n, "home_deg": _num(math.degrees(DEFAULT_POSITION[j]), 1),
                   "limits_deg": [_num(math.degrees(r.safety.limits[j][0] - DEFAULT_POSITION[j]), 1),
                                  _num(math.degrees(r.safety.limits[j][1] - DEFAULT_POSITION[j]), 1)],
                   "mouth": j == MOUTH_INDEX}
            if cal is not None:
                jc = cal.joints[j]
                row.update(id=jc.id, zero_raw=jc.zero_raw, direction=jc.direction)
            joints.append(row)
        return {
            "host": socket.gethostname(), "backend": self.backend, "started": self.started,
            "calib": self.calib_path, "joints": joints, "limits": LIMITS,
            "policy": {"walk": getattr(self.pset, "walk_file", None) if self.pset else "single policy",
                       "sitstand": bool(self.pset and self.pset.sitstand), "skills": skills},
            "sets": [{"label": ls.label, "path": ls.path, "active": i == self.bank.index,
                      "servos": getattr(ls.pset, "servos", ""), "description": getattr(ls.pset, "description", ""),
                      "action_scale": ls.tuning.action_scale}
                     for i, ls in enumerate(self.bank.sets)],
            "head_offset_deg": [round(math.degrees(v), 1) for v in self.head_offset],
            "control_hz": round(1.0 / self.io.control_dt),
            "tuning": {"action_scale": r.controller.tuning.action_scale},
            "on_fall": r.on_fall, "getup_timeout_s": r.getup_timeout_s,
            "token_required": self.token is not None,
            "mouth": {"servo": self.mouth_servo(), "range_deg": MOUTH_RANGE_DEG},
        }

    def log_event(self, level: str, message: str) -> None:
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "wall": time.time(), "level": level, "message": message})

    # -- control thread ------------------------------------------------------------------------
    def _track_events(self) -> None:
        r = self.runner
        now = {"state": r.state.value, "mode": r.mode if r.state is State.RUN else "", "refusal": r.refusal,
               "falls": r.falls_seen, "limped": r.fall_events}
        p = self._prev
        if p:
            if now["state"] != p["state"]:
                self.log_event("state", f"state {p['state']} -> {now['state']}")
            if now["mode"] != p["mode"] and now["mode"] and now["state"] == "run":
                self.log_event("mode", f"policy: {now['mode']}")
            if now["falls"] > p["falls"]:
                self.log_event("warn", "fall detected")
            if now["limped"] > p["limped"]:
                self.log_event("error", "fall -> torque off" + (f" ({r.refusal})" if r.refusal else ""))
            if now["refusal"] != p["refusal"] and now["refusal"]:
                self.log_event("warn", now["refusal"])
        self._prev = now

    def _build_frame(self) -> dict:
        r, io = self.runner, self.io
        s = r.last_sensors
        pos = tgt = vel = None
        pitch = roll = None
        gyro = None
        imu_ready = False
        if s is not None:
            rel = np.asarray(s.positions) - DEFAULT_POSITION
            pos = _arr(np.degrees(rel), 1)
            vel = _arr(np.degrees(s.velocities), 0)
            pitch, roll = pitch_roll(s.imu.gravity)
            gyro = _arr(np.degrees(s.imu.gyro), 0)
            imu_ready = bool(s.imu_ready)
        if r.last_targets is not None and r.state is not State.LIMP:
            tgt = _arr(np.degrees(np.asarray(r.last_targets) - DEFAULT_POSITION), 1)
        cur = getattr(io, "currents", None)
        volt = getattr(io, "voltages", None)
        temp = getattr(io, "temperatures", None)
        if volt is not None:
            v = np.asarray(volt, dtype=float).copy()
            v[MOUTH_INDEX] = np.nan
            v[v <= 0] = np.nan
            supply = {"min": _num(np.nanmin(v)) if np.isfinite(v).any() else None,
                      "mean": _num(np.nanmean(v)) if np.isfinite(v).any() else None}
        else:
            supply = {"min": None, "mean": None}
        if cur is not None:
            c = np.abs(np.asarray(cur, dtype=float))
            supply["current_a"] = _num(np.nansum(np.delete(c, MOUTH_INDEX)) / 1000.0)
        obs = r.controller.last_obs
        action_max = (_num(float(np.abs(obs[34:48]).max())) if obs is not None and r.state is State.RUN else None)
        tt = list(self.tick_times)
        dts = np.diff(tt) if len(tt) > 2 else np.array([])
        mv = None
        if r.move is not None:
            mv = {"name": r.move.skill.name, "remaining": _num(max(0.0, r.move.remaining)),
                  "duration": _num(r.move.skill.duration_s), "unwinding": r.move.unwinding}
        return {
            "seq": self.frame_seq, "wall": time.time(), "t": _num(io.now(), 2),
            "state": r.state.value, "mode": r.mode if r.state is State.RUN else "",
            "set": self.bank.active.label if self.bank.active is not None else None,
            "sit": r.sit, "busy": r.busy, "move": mv, "refusal": r.refusal,
            "falls": r.falls_seen, "limped": r.fall_events,
            "on_fall": r.on_fall, "getup_timeout_s": r.getup_timeout_s,
            "cmd": {**self.cmd, "age_s": _num(min(99.0, time.monotonic() - self.cmd_wall), 2)},
            "mouth_deg": _num(math.degrees(r.mouth_target), 1),
            "imu": {"ready": imu_ready, "pitch": _num(pitch, 1), "roll": _num(roll, 1), "gyro": gyro},
            "joints": {"pos": pos, "tgt": tgt, "vel": vel,
                       "cur": _arr(cur, 0) if cur is not None else None,
                       "volt": _arr(volt, 2) if volt is not None else None,
                       "temp": _arr(temp, 0) if temp is not None else None},
            "supply": supply, "action_max": action_max,
            "loop": {"hz": _num(1.0 / dts.mean(), 1) if dts.size else None,
                     "max_ms": _num(dts.max() * 1000.0, 1) if dts.size else None, "overruns": self.overruns},
            "recording": ({"on": True, "seconds": _num(time.time() - self.rec_started, 0),
                           "ticks": len(self.recording)} if self.recording is not None else {"on": False}),
            "clients": self.clients,
            "events_seq": self.event_seq,
        }

    def run(self) -> None:
        r, io = self.runner, self.io
        dt = io.control_dt
        next_frame = 0.0
        self.log_event("info", f"dashboard up ({self.backend})")
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            while True:
                try:
                    job = self.actions.get_nowait()
                except queue.Empty:
                    break
                job()
            if time.monotonic() - self.cmd_wall <= COMMAND_TTL_S:
                c = self.cmd
                r.set_command(Command(twist=(c["vx"], c["vy"], c["wz"]),
                                      head=tuple(self.head_offset[i] + math.radians(c["head"][i]) for i in range(4))),
                              io.now())
            r.tick()                              # paces itself on the robot (wait_tick)
            if self.realtime:
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
            now = time.monotonic()
            if self.tick_times and now - self.tick_times[-1] > 1.5 * dt:
                self.overruns += 1
            self.tick_times.append(now)
            self._track_events()
            if now >= next_frame:
                next_frame = now + 1.0 / FRAME_HZ
                frame = self._build_frame()
                with self.cond:
                    self.frame = frame
                    self.frame_seq += 1
                    self.cond.notify_all()

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()


def make_handler(dash: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "duck-dashboard"

        def log_message(self, fmt, *args):   # quiet: the terminal shows the robot, not HTTP
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(dash.info())
            elif path == "/api/logs":
                self._json(dash.list_logs())
            elif path == "/api/events":
                since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                self._json([e for e in dash.events if e["seq"] > since])
            elif path.startswith("/logs/"):
                name = path[len("/logs/"):]
                p = dash.log_dir / name
                if not LOG_NAME.match(name) or not p.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                self._send(200, p.read_bytes(), "application/octet-stream",
                           {"Content-Disposition": f'attachment; filename="{name}"'})
            elif path == "/api/stream":
                self._stream()
            else:
                self._json({"error": "not found"}, 404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            dash.clients += 1
            seen = -1
            try:
                while not dash.stop_event.is_set():
                    with dash.cond:
                        dash.cond.wait_for(lambda: dash.frame_seq != seen or dash.stop_event.is_set(), timeout=2.0)
                        frame, seen = dash.frame, dash.frame_seq
                    if frame is None:
                        self.wfile.write(b": waiting\n\n")
                    else:
                        self.wfile.write(b"data: " + json.dumps(frame, allow_nan=False).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                dash.clients -= 1

        def do_POST(self):
            path = urlparse(self.path).path
            # Read the body before any reply: an unread body on a keep-alive connection would be
            # parsed as the next request.
            n = int(self.headers.get("Content-Length") or 0)
            if n > 65536:
                self.close_connection = True
                self._json({"ok": False, "error": "body too large"}, 413)
                return
            raw = self.rfile.read(n) if n else b""
            if dash.token is not None and self.headers.get("X-Duck-Token") != dash.token:
                self._json({"ok": False, "error": "missing or wrong token: open the URL printed on the robot"}, 403)
                return
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                route = {"/api/command": dash.set_command, "/api/action": dash.action, "/api/mouth": dash.set_mouth,
                         "/api/settings": dash.settings, "/api/record": dash.record}.get(path)
                if route is None:
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                self._json(route(body))
            except (ValueError, TypeError, KeyError) as e:
                self._json({"ok": False, "error": f"bad request: {e}"}, 400)

    return Handler


def _urls(port: int, token: str | None) -> list[str]:
    q = f"/?token={token}" if token else "/"
    hosts = [f"{socket.gethostname()}.local"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        hosts.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return [f"http://{h}:{port}{q}" for h in hosts]


def main(argv=None):
    from .run import add_runner_args, build_runner
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runner_args(ap)
    ap.add_argument("--host", default="0.0.0.0", help="listen address (0.0.0.0 = every network interface)")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--token", default=None, help="fixed control token (default: a new random one each start)")
    ap.add_argument("--no-token", action="store_true", help="anyone who can reach the page may control the robot")
    ap.add_argument("--log-dir", default="logs", help="where recordings are written")
    args = ap.parse_args(argv)

    runner, io, bank, head = build_runner(args, ap)
    token = None if args.no_token else (args.token or secrets.token_urlsafe(9))
    dash = Dashboard(runner, io, bank, head, backend=args.backend, log_dir=args.log_dir,
                     token=token, realtime=args.backend == "sim", calib_path=args.calib)
    if args.backend == "sim":
        runner.enable()                       # nobody is holding a simulated duck
    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(dash))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("dashboard:", "  ".join(_urls(args.http_port, token)))
    print("Ctrl+C here: torque off and exit")
    try:
        dash.run()
    except KeyboardInterrupt:
        pass
    finally:
        dash.shutdown()
        if dash.recording:
            from .run import save_log
            name = time.strftime("dash_%Y%m%d-%H%M%S.npz")
            save_log(str(dash.log_dir / name), dash.recording)
            print(f"\nrecording -> {dash.log_dir / name}")
        runner.limp()
        io.close()
        server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
