"""Browser UI for the duck in simulation (viser / mjviser: the same stack as mjlab's viewer).

The 3D scene and its Scene / Visualization / Groups tabs are mjviser's (camera tracking,
contacts, forces, collision geoms...). The "Duck" tab drives the SAME Runner / Controller /
policy set the robot runs (duck_runtime), on SimIO (MuJoCo + BAM HD1910 + mass profile):
power, policy start/stop, velocity and head commands, sit/stand and every one-shot of the
set, pushes, and live trunk pitch / roll and action plots.

  uv run scripts/viser_duck.py --set policies                 # official set from the robot
  uv run scripts/viser_duck.py --policy policies/velstand.onnx
  uv run scripts/viser_duck.py --set policies --profile official   # original mass distribution
  -> open http://localhost:8080  (on a remote GPU box: ssh -L 8080:localhost:8080 <box>)

Pitch sign: + = nose down, - = leaning back (as in the robot logs).
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time
from collections import deque

import numpy as np
import viser
import viser.uplot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mjviser.scene import ViserMujocoScene  # noqa: E402

from duck_runtime.controller import Command, Controller, Policy, Tuning  # noqa: E402
from duck_runtime.policy_set import load_set  # noqa: E402
from duck_runtime.runner import Runner, State  # noqa: E402
from duck_runtime.sim_io import SimIO  # noqa: E402

HISTORY_S = 10.0
RENDER_HZ = 30.0
PLOT_HZ = 10.0
SPEEDS = {"1/4x": 0.25, "1/2x": 0.5, "1x": 1.0}
HEAD_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


SUPPLIES = {"ideal": dict(vin=7.6, vin_drop_gain=0.1, vin_min=6.0),
            "robot": dict(vin=7.5, vin_drop_gain=1.8, vin_min=3.0),
            "robot-full": dict(vin=8.4, vin_drop_gain=1.8, vin_min=3.0)}


def supply_kwargs(args) -> dict:
    return SUPPLIES[getattr(args, "supply", "ideal")]


def pitch_roll(g) -> tuple[float, float]:
    return (math.degrees(math.atan2(-g[0], -g[2])), math.degrees(math.atan2(g[1], -g[2])))


class DuckApp:
    def __init__(self, args):
        self.args = args
        if args.policy_set:
            self.pset = load_set(args.policy_set, walk=args.policy, sitstand=args.sitstand)
            self.walk = self.pset.walk
        elif args.policy:
            self.pset, self.walk = None, Policy(args.policy)
        else:
            raise SystemExit("--set or --policy is required")
        self.tuning = Tuning.official()
        if args.action_scale is not None:
            self.tuning.action_scale = args.action_scale

        self.io = SimIO(mass_profile=None if args.profile == "official" else args.profile, servo="hd1910",
                        seed=args.seed, **supply_kwargs(args))
        self.actions: queue.Queue = queue.Queue()   # GUI threads -> sim loop
        self.speed = 1.0
        self.message = ""
        self._new_runner()

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.gui.configure_theme(control_width="large")
        self.scene = ViserMujocoScene(self.server, self.io.model, num_envs=1)
        self._build_gui()

    # -- simulation -------------------------------------------------------------------------
    def _new_runner(self) -> None:
        self.io.reset(joint_noise=0.0)
        self.runner = Runner(self.io, Controller(self.walk, self.tuning), policies=self.pset,
                             on_fall=self.args.on_fall)
        self.runner.enable()                        # powered at HOME, like run.py --backend sim
        self._autostart_pending = True
        n = int(HISTORY_S / self.io.control_dt)
        self.hist_t, self.hist_pitch, self.hist_roll, self.hist_act = (deque(maxlen=n) for _ in range(4))

    def _do(self, fn, label: str | None = None):
        def cb(_=None):
            self.actions.put((fn, label))
        return cb

    def _report(self, result, ok: str):
        self.message = ok if result is None else f"refused: {result}"

    # -- GUI --------------------------------------------------------------------------------
    def _build_gui(self) -> None:
        gui = self.server.gui
        tabs = self.scene.create_visualization_gui(camera_distance=0.9, camera_azimuth=135.0, camera_elevation=15.0)
        with tabs.add_tab("Duck", icon=viser.Icon.ROBOT):
            self.status = gui.add_html("")

            with gui.add_folder("Power / policy"):
                b = gui.add_button_group("Runner", options=["1 enable", "2 start", "3 stop", "0 limp"])

                @b.on_click
                def _(ev) -> None:
                    v = ev.target.value
                    r = self.runner
                    fn = {"1 enable": r_enable, "2 start": r_start, "3 stop": lambda: r_call("stop"),
                          "0 limp": lambda: r_call("limp")}[v]
                    self.actions.put((fn, None))

                def r_call(name):
                    getattr(self.runner, name)()
                    self.message = name

                def r_enable():
                    if self.runner.state is State.LIMP:
                        self.runner.enable()
                    self.message = "enable"

                def r_start():
                    ok = self.runner.start()
                    self.message = "policy started" if ok else f"refused: {self.runner.refusal or self.runner.state.value}"

                reset = gui.add_button("Reset simulation", icon=viser.Icon.REFRESH)
                reset.on_click(self._do(lambda: (self._new_runner(), setattr(self, "message", "reset"))))
                speed = gui.add_button_group("Speed", options=list(SPEEDS))

                @speed.on_click
                def _(ev) -> None:
                    self.speed = SPEEDS[ev.target.value]
                self.autostart = gui.add_checkbox(
                    "Auto-start policy", initial_value=True,
                    hint="Start the policy as soon as the ramp to HOME ends (a simulated duck held at HOME "
                         "with the policy off slowly tips over)")
                fall = gui.add_dropdown("On fall", options=["limp", "keep", "getup"], initial_value=self.args.on_fall,
                                        hint="limp = the robot runtime's rule; keep = robotd's default")

                @fall.on_update
                def _(_) -> None:
                    self.actions.put((lambda: setattr(self.runner, "on_fall", fall.value), None))

            with gui.add_folder("Command"):
                self.vx = gui.add_slider("vx (m/s)", min=-0.3, max=0.3, step=0.05, initial_value=0.0)
                self.vy = gui.add_slider("vy (m/s)", min=-0.2, max=0.2, step=0.05, initial_value=0.0)
                self.wz = gui.add_slider("wz (rad/s)", min=-1.0, max=1.0, step=0.1, initial_value=0.0)
                self.head = [gui.add_slider(f"{n} (deg)", min=-60.0, max=60.0, step=1.0, initial_value=0.0)
                             for n in HEAD_NAMES]
                zero = gui.add_button("Zero command")

                @zero.on_click
                def _(_) -> None:
                    for s in (self.vx, self.vy, self.wz, *self.head):
                        s.value = 0.0

            with gui.add_folder("Moves"):
                if self.pset is not None and self.pset.sitstand is not None:
                    sit = gui.add_button("Sit / stand (x)")
                    sit.on_click(self._do(lambda: self._report(self.runner.sit_toggle(), f"sit/stand: {self.runner.sit}")))
                for name, sk in (self.pset.skills.items() if self.pset else []):
                    btn = gui.add_button(f"{name} ({sk.duration_s:.1f} s)")
                    btn.on_click(self._do(lambda n=name: self._report(self.runner.trigger(n), n)))
                if self.pset is None or (not self.pset.skills and self.pset.sitstand is None):
                    gui.add_markdown("_Start with `--set policies` for sit/stand and the one-shots._")

            with gui.add_folder("Push", expand_by_default=False):
                self.push_x = gui.add_slider("dvx (m/s, + = back)", min=-0.6, max=0.6, step=0.05, initial_value=0.3)
                self.push_y = gui.add_slider("dvy (m/s)", min=-0.6, max=0.6, step=0.05, initial_value=0.0)
                push = gui.add_button("Push", icon=viser.Icon.HAND_STOP)
                push.on_click(self._do(lambda: (self.io.push(self.push_x.value, self.push_y.value),
                                                setattr(self, "message", "push"))))

            with gui.add_folder("Plots"):
                empty = np.zeros(0)
                self.plot_att = gui.add_uplot(
                    (empty, empty, empty),
                    (viser.uplot.Series(label="t (s)"),
                     viser.uplot.Series(label="pitch", stroke="#e4572e", width=2),
                     viser.uplot.Series(label="roll", stroke="#4c9bd6", width=2)),
                    title="Trunk attitude (deg)", aspect=1.6,
                    scales={"x": viser.uplot.Scale(time=False, auto=True), "y": viser.uplot.Scale(auto=True)})
                self.plot_act = gui.add_uplot(
                    (empty, empty),
                    (viser.uplot.Series(label="t (s)"),
                     viser.uplot.Series(label="max |action|", stroke="#7a5195", width=2)),
                    title="Policy output", aspect=2.2,
                    scales={"x": viser.uplot.Scale(time=False, auto=True), "y": viser.uplot.Scale(auto=True)})

    # -- loop -------------------------------------------------------------------------------
    def _status_html(self, s) -> str:
        r = self.runner
        p, rl = pitch_roll(s.imu.gravity)
        z = self.io.trunk_pos()[2]
        state = r.state.value + (f" · {r.mode}" if r.state is State.RUN else "")
        return f"""<div style="font-size:0.9em;line-height:1.45;padding:0 1em 0.5em 1em;">
          <b>State</b> {state}<br/>
          <b>Pitch</b> {p:+.1f}°  <b>Roll</b> {rl:+.1f}°  <b>Height</b> {z * 100:.1f} cm  <b>Supply</b> {self.io.voltages[0]:.1f} V<br/>
          <b>Falls</b> {r.falls_seen} (limped {r.fall_events})  <b>Sim</b> {self.io.now():.1f} s  <b>Speed</b> {self.speed:g}x<br/>
          <span style="color:#888;">{self.message}</span></div>"""

    def run(self) -> None:
        print(f"open http://localhost:{self.args.port}  (Ctrl+C to quit)")
        dt = self.io.control_dt
        next_render = next_plot = 0.0
        budget, last = 0.0, time.perf_counter()
        while True:
            now = time.perf_counter()
            budget += (now - last) * self.speed
            last = now
            while not self.actions.empty():
                fn, _ = self.actions.get()
                try:
                    fn()
                except Exception as e:  # a GUI action must never kill the loop
                    self.message = f"error: {e}"
            steps = 0
            while budget >= dt and steps < 5:
                head = tuple(math.radians(h.value) for h in self.head)
                self.runner.set_command(Command(twist=(self.vx.value, self.vy.value, self.wz.value), head=head),
                                        self.io.now())
                s = self.runner.tick()
                if self._autostart_pending and self.runner.state is State.HOLD:
                    self._autostart_pending = False
                    if self.autostart.value and self.runner.start():
                        self.message = "policy started (auto)"
                g = s.imu.gravity
                p, rl = pitch_roll(g)
                self.hist_t.append(self.io.now())
                self.hist_pitch.append(p)
                self.hist_roll.append(rl)
                obs = self.runner.controller.last_obs
                self.hist_act.append(float(np.abs(obs[34:48]).max()) if self.runner.state is State.RUN and obs is not None else 0.0)
                budget -= dt
                steps += 1
            if steps == 5:
                budget = 0.0                         # can't keep up: drop time rather than spiral
            if now >= next_render:
                next_render = now + 1.0 / RENDER_HZ
                self.scene.update_from_mjdata(self.io.data)
            if now >= next_plot and self.hist_t:
                next_plot = now + 1.0 / PLOT_HZ
                t = np.asarray(self.hist_t) - self.hist_t[-1]
                self.plot_att.data = (t, np.asarray(self.hist_pitch), np.asarray(self.hist_roll))
                self.plot_act.data = (t, np.asarray(self.hist_act))
                self.status.content = self._status_html(self.io.read())
            time.sleep(0.002)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="policy_set", help="directory with manifest.json + .onnx (from the robot)")
    ap.add_argument("--policy", help="gait ONNX; with --set overrides its walk slot")
    ap.add_argument("--sitstand", help="with --set: another sit/stand network")
    ap.add_argument("--supply", choices=["ideal", "robot", "robot-full"], default="ideal",
                    help="battery sag model: ideal = training-like (7.6 V, floor 6 V); robot = this robot's "
                         "measured sag (set6.npz: 7.5 V idle, ~4.6 V in a rise, 3-4 V in a kick); "
                         "robot-full = same wiring with a fully charged 8.4 V pack")
    ap.add_argument("--profile", default="ours_835g", help="mass profile: ours_835g | ours_813g | official | ...")
    ap.add_argument("--on-fall", choices=["limp", "keep", "getup"], default="limp")
    ap.add_argument("--action-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)
    app = DuckApp(args)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.server.stop()


if __name__ == "__main__":
    main()
