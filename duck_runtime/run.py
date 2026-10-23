"""Drive the duck from a terminal. Same loop for the simulated and the real body.

  # rehearse on a laptop (MuJoCo viewer if a display is available)
  python -m duck_runtime.run --backend sim --policy velstand.onnx --viewer
  # this robot (ESP32-S3 bridge on /dev/ttyACM0 + Pi I2C IMU), after duck_runtime.tools calibration
  sudo systemctl stop microduck-gamepad.service
  python -m duck_runtime.run --calib calib.json --policy velstand.onnx

  # the official set (sit/stand, ground pick, kicks, roulade): the Hub repo in policies/
  python -m duck_runtime.run --calib calib.json --set policies

Keys:  1 enable (torque on, ease to HOME)   2 start policy   3 stop (back to HOME)
       0 limp (torque off)                  w/s vx +-0.05    z/c vy +-0.05
       a/d yaw +-0.2                        space zero command
  with --set:  x sit / stand    g ground pick    k / l kick left / right    r roulade
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

import numpy as np

from .controller import Command, Controller, Policy, Tuning
from .runner import Runner, State, TickLog
from .safety import Safety

VX_MAX, VY_MAX, WZ_MAX = 0.3, 0.2, 1.0


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["bridge", "feetech", "sim"], default="bridge")
    ap.add_argument("--policy", help="61->14 ONNX gait (e.g. official velstand.onnx); with --set: overrides its walk slot")
    ap.add_argument("--set", dest="policy_set", help="directory with the official set's manifest.json + .onnx "
                    "(sit/stand, ground pick, kicks, roulade)")
    ap.add_argument("--rise-s", type=float, default=None,
                    help="with --set: seconds the sit/stand network drives the rise (manifest: 1.0)")
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
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--keep-on-fall", action="store_true",
                    help="keep driving after a fall so velstand v8 gets the robot up (official default); "
                         "also lets 2 start the policy with the robot on the floor")
    ap.add_argument("--getup-timeout", type=float, default=5.0,
                    help="with --keep-on-fall: limp if still down after this many seconds (0 = never)")
    ap.add_argument("--log", help="write an .npz log of every tick on exit")
    ap.add_argument("--head-cmd", default="0,0,0,0",
                    help="head pose command, degrees from HOME: neck_pitch,head_pitch,head_yaw,head_roll "
                         "(also used for the stand pose before the policy starts)")
    args = ap.parse_args(argv)

    tuning = Tuning.official() if args.tuning == "official" else Tuning.as_trained()
    if args.action_scale is not None:
        tuning.action_scale = args.action_scale

    if args.backend == "sim":
        from .sim_io import SimIO
        io = SimIO()
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

    log: list[TickLog] | None = [] if args.log else None
    head = tuple(math.radians(float(v)) for v in args.head_cmd.split(","))
    if len(head) != 4:
        ap.error("--head-cmd needs 4 values")
    pset, skill_keys = None, {}
    if args.policy_set:
        from .policy_set import bind_keys, load_set
        pset = load_set(args.policy_set, walk=args.policy, sitstand=args.sitstand)
        skill_keys = bind_keys(pset.skills)
        if args.rise_s is not None and pset.sitstand is not None:
            pset.sitstand.rise_s = args.rise_s
        walk = pset.walk
        print(f"policy set {args.policy_set}: walk {pset.walk_file}, sit/stand "
              f"{'x' if pset.sitstand else '-'}, skills "
              + (", ".join(f"{k}={n} ({pset.skills[n].duration_s:.1f} s)" for k, n in skill_keys.items()) or "-"))
    elif args.policy:
        walk = Policy(args.policy)
    else:
        ap.error("--policy or --set is required")
    runner = Runner(io, Controller(walk, tuning), safety,
                    on_fall="keep" if args.keep_on_fall else "limp", log=log, head_offset=head,
                    policies=pset, getup_timeout_s=args.getup_timeout or None)
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
                    elif k == "x" and pset is not None:
                        why = runner.sit_toggle()
                        print(f"\n[{'sit/stand' if why is None else 'refused'}] {why or runner.sit}")
                    elif k in skill_keys:
                        vx = vy = wz = 0.0           # a move starts from standing; walk resumes at zero
                        why = runner.trigger(skill_keys[k])
                        print(f"\n[{'skill' if why is None else 'refused'}] {why or skill_keys[k]}")
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
            np.savez_compressed(args.log, t=[r.t for r in log], state=[r.state for r in log],
                                obs=np.array([np.full(61, np.nan) if r.obs is None else r.obs for r in log]),
                                targets=np.array([np.full(15, np.nan) if r.targets is None else r.targets for r in log]),
                                positions=np.array([r.positions for r in log]),
                                gravity=np.array([r.gravity for r in log]), gyro=np.array([r.gyro for r in log]),
                                currents=np.array([np.full(15, np.nan) if r.currents is None else r.currents for r in log]),
                                mode=[r.mode for r in log])
            print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main(sys.argv[1:])
