"""Drive the duck from a terminal. Same loop for the simulated and the real body.

  # rehearse on a laptop (MuJoCo viewer if a display is available)
  python -m duck_runtime.run --backend sim --policy velstand.onnx --viewer
  # on the robot (Pi), after duck_runtime.tools calibration
  python -m duck_runtime.run --backend feetech --port /dev/ttyUSB0 --calib calib.json --policy velstand.onnx

Keys:  1 enable (torque on, ease to HOME)   2 start policy   3 stop (back to HOME)
       0 limp (torque off)                  w/s vx +-0.05    z/c vy +-0.05
       a/d yaw +-0.2                        space zero command
       Ctrl+C: limp and exit.  The policy needs >= ~0.2 m/s to start stepping.
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--backend", choices=["sim", "feetech"], required=True)
    ap.add_argument("--policy", required=True, help="61->14 ONNX (e.g. official velstand.onnx)")
    ap.add_argument("--tuning", choices=["official", "trained"], default="official",
                    help="official: scale 0.9 + low-pass 0.5/0.7 (robotd defaults); trained: 1.0, no filter")
    ap.add_argument("--action-scale", type=float, default=None, help="override, e.g. 0.7 for a timid first run")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--echo", action="store_true")
    ap.add_argument("--calib")
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--keep-on-fall", action="store_true", help="keep driving after a fall (official default)")
    ap.add_argument("--log", help="write an .npz log of every tick on exit")
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
        from .feetech import FeetechIO
        if not args.calib:
            ap.error("--calib is required on the robot")
        cal = Calibration.load(args.calib)
        cal.validate(require_calibrated=True)
        io = FeetechIO(args.port, cal, echo=args.echo)
        safety = Safety(cal.limits())

    log: list[TickLog] | None = [] if args.log else None
    runner = Runner(io, Controller(Policy(args.policy), tuning), safety,
                    on_fall="keep" if args.keep_on_fall else "limp", log=log)
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
                    if k == "1": runner.enable()
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
                runner.set_command(Command(twist=(vx, vy, wz)), io.now())
                runner.tick()
                if viewer is not None:
                    viewer.sync()
                now = time.monotonic()
                if now - last_print > 0.5:
                    last_print = now
                    print(f"\r[{runner.state.value:4s}] cmd vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}  "
                          f"falls {runner.fall_events}   ", end="", flush=True)
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
                                gravity=np.array([r.gravity for r in log]), gyro=np.array([r.gyro for r in log]))
            print(f"\nlog -> {args.log}")


if __name__ == "__main__":
    main(sys.argv[1:])
