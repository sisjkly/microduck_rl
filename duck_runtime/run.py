"""Drive the duck from a terminal. Same loop for the simulated and the real body.

  # rehearse on a laptop (MuJoCo viewer if a display is available)
  python -m duck_runtime.run --backend sim --policy velstand.onnx --viewer
  # this robot (ESP32-S3 bridge on /dev/ttyACM0 + Pi I2C IMU), after duck_runtime.tools calibration
  sudo systemctl stop microduck-gamepad.service
  python -m duck_runtime.run --calib calib.json --policy velstand.onnx

  # the official set (sit/stand, ground pick, kicks, roulade): the Hub repo in policies/
  python -m duck_runtime.run --calib calib.json --set policies
  # two complete sets, switch with p (official first = active at start; kissqy's set from
  # duck_runtime.fetch_set kissqy). Each set walks, moves and gets up with its own networks only.
  python -m duck_runtime.run --calib calib.json --set policies --set policies_hd1910

Keys:  1 enable (torque on, ease to HOME)   2 start policy   3 stop (back to HOME)
       0 limp (torque off)                  w/s vx +-0.05    z/c vy +-0.05
       a/d yaw +-0.2                        space zero command
  with --set:  x sit / stand    g ground pick    k / l kick left / right    r roulade
       p next policy set (several --set; only with the policy stopped: 3 first)
       m open / close the mouth (when calib.json gives the mouth a servo: tools mouth)
       Ctrl+C: limp and exit.  The policy needs >= ~0.2 m/s to start stepping.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, replace

import numpy as np

from .controller import Command, Controller, Policy, Tuning
from .runner import Runner, State, TickLog
from .safety import Safety

VX_MAX, VY_MAX, WZ_MAX = 0.3, 0.2, 1.0
MOUTH_OPEN = math.radians(25.0)


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self) -> str:
        # os.read, not sys.stdin.read: the text wrapper buffers a whole chunk, after which
        # select() no longer reports the remaining keys.
        out = b""
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 1024)
            if not chunk:
                break
            out += chunk
        return out.decode(errors="ignore")


def add_runner_args(ap: argparse.ArgumentParser) -> None:
    """Robot / policy / safety options shared by the terminal driver and the dashboard."""
    ap.add_argument("--backend", choices=["bridge", "feetech", "sim"], default="bridge")
    ap.add_argument("--policy", help="61->14 ONNX gait (e.g. official velstand.onnx); with one --set: overrides "
                    "its walk slot (refused with several --set: sets are never mixed)")
    ap.add_argument("--set", dest="policy_set", action="append",
                    help="directory with a set's manifest.json + .onnx (sit/stand, ground pick, kicks, roulade); "
                         "give it twice (e.g. --set policies --set policies_hd1910) to switch between two "
                         "complete, separate sets: key p in the terminal, the selector on the dashboard; the "
                         "first one starts active")
    ap.add_argument("--rise-s", type=float, default=None,
                    help="with --set: seconds the sit/stand network drives the rise (manifest: 1.0)")
    ap.add_argument("--skill-duration", action="append", default=[], metavar="NAME=SECONDS",
                    help="with --set: how long a move's network drives before the gait takes over, e.g. "
                         "roulade=3 (manifest: roulade 1.0, kicks 0.5, ground pick 2.8); repeatable, every set")
    ap.add_argument("--sitstand", help="with --set: another sit/stand network (twist vx = sit flag), "
                    "e.g. the got-a-talent reference")
    ap.add_argument("--tuning", choices=["official", "trained"], default="official",
                    help="official: scale 0.9 + low-pass 0.5/0.7 (robotd defaults); trained: 1.0, no filter")
    ap.add_argument("--action-scale", type=float, default=None, help="override, e.g. 0.7 for a timid first run")
    ap.add_argument("--port", default=None, help="default /dev/ttyACM0 (bridge) or /dev/ttyUSB0 (feetech)")
    ap.add_argument("--i2c-bus", type=int, default=1, help="bridge: Pi I2C bus of the LSM6DSV16X")
    ap.add_argument("--echo", choices=["auto", "yes", "no"], default="auto", help="feetech only")
    ap.add_argument("--imu-port", default=None, help="feetech only: ESP32 IMU link port")
    ap.add_argument("--calib")
    ap.add_argument("--on-fall", choices=["limp", "keep", "getup"], default=None,
                    help="limp: torque off (default). keep: keep driving, velstand v8 gets up. "
                         "getup: robotd's default limp-fall: go limp as the fall starts, land soft, "
                         "ease into the stand pose, then the policy stands up. keep/getup also let "
                         "2 start the policy with the robot on the floor")
    ap.add_argument("--keep-on-fall", action="store_true", help="same as --on-fall keep")
    ap.add_argument("--getup-timeout", type=float, default=5.0,
                    help="with keep/getup: limp if still down after this many seconds (0 = never)")
    ap.add_argument("--supply", choices=["ideal", "robot", "robot-full"], default="ideal",
                    help="sim only: battery sag model (robot = this robot's measured sag)")
    ap.add_argument("--head-cmd", default="0,0,0,0",
                    help="head pose command, degrees from HOME: neck_pitch,head_pitch,head_yaw,head_roll "
                         "(also used for the stand pose before the policy starts)")


@dataclass
class LoadedSet:
    """One policy set and the gait tuning it runs with."""
    label: str
    path: str
    pset: object                    # PolicySet
    keys: dict                      # key -> skill name
    tuning: Tuning

    def summary(self) -> str:
        p = self.pset
        return (f"{self.label}: walk {p.walk_file} (scale {self.tuning.action_scale:g}, low-pass "
                f"{self.tuning.head_lowpass}/{self.tuning.legs_lowpass}), sit/stand {'x' if p.sitstand else '-'}, "
                f"skills " + (", ".join(f"{k}={n} ({p.skills[n].duration_s:g} s)" for k, n in self.keys.items()) or "-"))


class SetBank:
    """The sets given with --set; ``active`` is the one the runner drives."""

    def __init__(self, sets: list[LoadedSet]):
        self.sets, self.index = sets, 0

    @property
    def active(self) -> LoadedSet | None:
        return self.sets[self.index] if self.sets else None

    def find(self, label: str) -> int | None:
        for i, ls in enumerate(self.sets):
            if ls.label == label:
                return i
        return None

    def switch(self, runner: Runner, index: int) -> str | None:
        """Make set ``index`` active. None = done, else the refusal (policy running)."""
        if not 0 <= index < len(self.sets):
            return f"no set #{index}"
        ls = self.sets[index]
        why = runner.swap_policies(ls.pset, ls.tuning)
        if why is None:
            self.index = index
        return why


def _set_label(path: str, pset) -> str:
    base = os.path.basename(os.path.normpath(path)) or path
    return f"{base} ({pset.servos})" if getattr(pset, "servos", "") else base


def parse_skill_durations(items: list[str], ap: argparse.ArgumentParser) -> dict[str, float]:
    """["roulade=3", ...] -> {"roulade": 3.0}. The window of a phase-encoded move (ground pick)
    is period * end_phase in training: lengthening it runs the phase past its trained end."""
    out = {}
    for item in items:
        name, _, sec = item.partition("=")
        try:
            v = float(sec)
        except ValueError:
            v = float("nan")
        if not name or not (0.1 <= v <= 10.0):
            ap.error(f"--skill-duration {item!r}: want NAME=SECONDS with 0.1..10 s, e.g. roulade=3")
        out[name.strip()] = v
    return out


def load_sets(args, ap: argparse.ArgumentParser, base_tuning: Tuning) -> SetBank:
    from .policy_set import apply_walk_tuning, bind_keys, load_set
    paths = args.policy_set or []
    if len(paths) > 1:
        # Two sets = two complete, separate robots-worth of networks (official / kissqy): each one
        # walks, sits, moves and gets up with its own networks and its own deployment settings.
        mixed = [f for f, v in (("--policy", args.policy), ("--sitstand", args.sitstand),
                                ("--action-scale", args.action_scale), ("--tuning trained", args.tuning != "official"))
                 if v]
        if mixed:
            ap.error(f"{', '.join(mixed)}: not with several --set (each set runs only its own networks and "
                     "settings); load one --set to try an override")
    durations = parse_skill_durations(getattr(args, "skill_duration", None) or [], ap)
    sets: list[LoadedSet] = []
    for i, path in enumerate(paths):
        pset = load_set(path, walk=args.policy, sitstand=args.sitstand)
        if args.rise_s is not None and pset.sitstand is not None:
            pset.sitstand.rise_s = args.rise_s
        for name, sec in durations.items():
            if name in pset.skills:
                pset.skills[name].duration_s = sec
        tuning = replace(base_tuning)
        if args.tuning == "official":       # the set's own gait settings (kissqy: 1.0, 0.5 / 0.7)
            apply_walk_tuning(pset, tuning, args.action_scale)
        label = _set_label(path, pset)
        if any(s.label == label for s in sets):
            label = f"{label} #{i + 1}"
        sets.append(LoadedSet(label, path, pset, bind_keys(pset.skills), tuning))
    unknown = [n for n in durations if not any(n in ls.pset.skills for ls in sets)]
    if unknown:
        ap.error(f"--skill-duration: no move named {unknown} in the loaded sets")
    return SetBank(sets)


def build_runner(args, ap: argparse.ArgumentParser):
    """-> (runner, io, set bank, head offset) from parsed args. ``bank.active`` is None with --policy only."""
    tuning = Tuning.official() if args.tuning == "official" else Tuning.as_trained()
    if args.action_scale is not None:
        tuning.action_scale = args.action_scale

    if args.backend == "sim":
        from .sim_io import SUPPLY_PRESETS, SimIO
        io = SimIO(**SUPPLY_PRESETS[getattr(args, "supply", "ideal")])
        safety = Safety()
    else:
        from .calibration import Calibration
        from .tools import open_io
        if not args.calib:
            ap.error("--calib is required on the robot")
        cal = Calibration.load(args.calib)
        cal.validate(require_calibrated=True)
        io = open_io(args, cal)
        safety = Safety(cal.limits())

    log: list[TickLog] | None = [] if getattr(args, "log", None) else None
    head = tuple(math.radians(float(v)) for v in args.head_cmd.split(","))
    if len(head) != 4:
        ap.error("--head-cmd needs 4 values")
    bank = load_sets(args, ap, tuning)
    if bank.active is not None:
        for i, ls in enumerate(bank.sets):
            print(("* " if i == 0 else "  ") + ls.summary())
        walk, tuning, pset = bank.active.pset.walk, bank.active.tuning, bank.active.pset
    elif args.policy:
        walk, pset = Policy(args.policy), None
    else:
        ap.error("--policy or --set is required")
    runner = Runner(io, Controller(walk, tuning), safety,
                    on_fall=args.on_fall or ("keep" if args.keep_on_fall else "limp"), log=log, head_offset=head,
                    policies=pset, getup_timeout_s=args.getup_timeout or None)
    return runner, io, bank, head


def save_log(path: str, log: list[TickLog]) -> None:
    np.savez_compressed(path, t=[r.t for r in log], state=[r.state for r in log],
                        obs=np.array([np.full(61, np.nan) if r.obs is None else r.obs for r in log]),
                        targets=np.array([np.full(15, np.nan) if r.targets is None else r.targets for r in log]),
                        positions=np.array([r.positions for r in log]),
                        gravity=np.array([r.gravity for r in log]), gyro=np.array([r.gyro for r in log]),
                        currents=np.array([np.full(15, np.nan) if r.currents is None else r.currents for r in log]),
                        mode=[r.mode for r in log],
                        voltages=np.array([np.full(15, np.nan) if r.voltages is None else r.voltages for r in log]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runner_args(ap)
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--log", help="write an .npz log of every tick on exit")
    args = ap.parse_args(argv)
    runner, io, bank, head = build_runner(args, ap)
    log = runner.log
    viewer = None
    if args.viewer and args.backend == "sim" and os.environ.get("DISPLAY"):
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(io.model, io.data)

    vx = vy = wz = 0.0
    last_print = 0.0
    print(__doc__)
    if args.backend == "sim":
        # A simulated duck has nobody holding it: start it powered at HOME instead of
        # letting it fold onto the floor while you reach for the keyboard.
        runner.enable()
    try:
        with Keys() as keys:
            while viewer is None or viewer.is_running():
                t0 = time.monotonic()
                for k in keys.poll():
                    if k == "1":
                        try:
                            runner.enable()
                        except RuntimeError as e:  # e.g. a latched bridge fault: say why, stay limp
                            runner.limp()
                            print(f"\n[refused] enable: {e}")
                    elif k == "2" and not runner.start(): print(f"\n[refused] start: {runner.refusal or runner.state.value}")
                    elif k == "3": runner.stop()
                    elif k == "0": runner.limp()
                    elif k == "w": vx = min(VX_MAX, vx + 0.05)
                    elif k == "s": vx = max(-VX_MAX, vx - 0.05)
                    elif k == "z": vy = min(VY_MAX, vy + 0.05)
                    elif k == "c": vy = max(-VY_MAX, vy - 0.05)
                    elif k == "a": wz = min(WZ_MAX, wz + 0.2)
                    elif k == "d": wz = max(-WZ_MAX, wz - 0.2)
                    elif k == " ": vx = vy = wz = 0.0
                    elif k == "m":
                        from .contract import MOUTH_CLOSED
                        runner.mouth_target = MOUTH_OPEN if runner.mouth_target <= MOUTH_CLOSED + 1e-6 else MOUTH_CLOSED
                    elif k == "x" and bank.active is not None:
                        why = runner.sit_toggle()
                        print(f"\n[{'sit/stand' if why is None else 'refused'}] {why or runner.sit}")
                    elif k == "p" and len(bank.sets) > 1:
                        why = bank.switch(runner, (bank.index + 1) % len(bank.sets))
                        print(f"\n[{'set' if why is None else 'refused'}] {why or bank.active.summary()}")
                    elif bank.active is not None and k in bank.active.keys:
                        vx = vy = wz = 0.0           # a move starts from standing; walk resumes at zero
                        name = bank.active.keys[k]
                        why = runner.trigger(name)
                        print(f"\n[{'skill' if why is None else 'refused'}] {why or name}")
                runner.set_command(Command(twist=(vx, vy, wz), head=head), io.now())
                runner.tick()
                if viewer is not None:
                    viewer.sync()
                now = time.monotonic()
                if now - last_print > 0.5:
                    last_print = now
                    print(f"\r[{runner.state.value:4s} {runner.mode:10s}] cmd vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}  "
                          f"falls {runner.falls_seen}  {runner.refusal}   ", end="", flush=True)
                if args.backend == "sim":  # real time for a human at the keyboard
                    time.sleep(max(0.0, io.control_dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        runner.limp()
        io.close()
        if log:
            save_log(args.log, log)
            print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main(sys.argv[1:])
