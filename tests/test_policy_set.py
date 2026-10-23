"""Policy set: manifest parsing and the per-tick network choice (robotd control.rs semantics)."""

import json
import math

import numpy as np
import pytest

from duck_runtime.contract import DEFAULT_POSITION, HEAD_JOINTS, POLICY_JOINTS
from duck_runtime.controller import Command, Controller, Tuning
from duck_runtime.imu import ImuData
from duck_runtime.io import RobotIO, Sensors
from duck_runtime.policy_set import bind_keys, load_set
from duck_runtime.runner import Runner, State

# The Hub manifest of pollen-robotics/microduck-policies (2026-10-12), verbatim structure.
OFFICIAL_MANIFEST = {
    "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
    "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
    "policies": [
        {"file": "alpha_walking.onnx", "kind": "perpetual"},
        {"file": "velstand.onnx", "kind": "perpetual", "slot": "walk", "entry_pose": "standing"},
        {"file": "alpha_stand.onnx", "kind": "perpetual"},
        {"file": "roller.onnx", "kind": "perpetual", "mode": "roller", "action_scale": 0.8},
        {"file": "alpha_sitstand.onnx", "name": "sitstand", "kind": "scripted",
         "command": {"encoding": "posture_flag", "slot": "twist.vx", "sit": 1.0, "stand": 0.0, "idle": [0.0, 0.0, 0.0]},
         "ramp_s": 2.0, "unwind_s": 1.0},
        {"file": "alpha_ground_pick.onnx", "name": "ground_pick", "kind": "episodic", "duration_s": 2.8,
         "command": {"encoding": "phase", "slots": "twist.vx,twist.vy", "period_s": 4.0, "end_phase": 0.7}},
        {"file": "roller_crouch.onnx", "name": "crouch", "kind": "episodic", "duration_s": 3.5, "mode": "roller",
         "action_scale": 0.8,
         "command": {"encoding": "phase", "slots": "twist.vx,twist.vy", "period_s": 5.0, "end_phase": 0.7}},
        {"file": "roulade.onnx", "kind": "episodic", "duration_s": 1.0, "chain": True},
        {"file": "ball_kick_left.onnx", "name": "kick_left", "kind": "episodic", "duration_s": 0.5},
        {"file": "ball_kick_right.onnx", "name": "kick_right", "kind": "episodic", "duration_s": 0.5},
    ],
}


class Recorder:
    """A fake network: a constant action, and the commands it was asked with."""

    def __init__(self, name, value):
        self.name, self.value, self.seen = name, value, []

    def __call__(self, obs):
        self.seen.append(obs[48:61].copy())
        return np.full(14, self.value, dtype=np.float32)


def make_set(tmp_path, files=None, manifest=OFFICIAL_MANIFEST):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    names = files or [e["file"] for e in manifest["policies"]]
    for f in names:
        (tmp_path / f).write_bytes(b"")
    nets = {}

    def loader(path):
        name = path.split("/")[-1].removesuffix(".onnx")
        nets[name] = Recorder(name, 0.01 * (len(nets) + 1))
        return nets[name]
    return loader, nets


def test_official_manifest(tmp_path):
    loader, nets = make_set(tmp_path)
    s = load_set(str(tmp_path), loader=loader)
    assert s.walk_file == "velstand.onnx" and s.walk is nets["velstand"]
    assert s.sitstand.policy is nets["alpha_sitstand"]
    assert (s.sitstand.ramp_s, s.sitstand.rise_s, s.sitstand.action_scale) == (2.0, 1.0, 1.0)
    assert list(s.skills) == ["ground_pick", "roulade", "kick_left", "kick_right"]  # roller entries skipped
    gp = s.skills["ground_pick"]
    assert gp.encoding == "phase" and gp.duration_s == pytest.approx(2.8) and gp.action_scale == 1.0
    assert s.skills["kick_left"].duration_s == 0.5 and s.skills["kick_left"].action_scale is None
    assert s.skills["roulade"].duration_s == 1.0
    assert "roller" not in nets and "alpha_walking" not in nets
    assert bind_keys(s.skills) == {"g": "ground_pick", "r": "roulade", "k": "kick_left", "l": "kick_right"}


def test_partial_download_and_overrides(tmp_path):
    loader, nets = make_set(tmp_path, ["velstand.onnx", "ball_kick_left.onnx", "sprint.onnx",
                                       "got_a_talent.onnx"])
    s = load_set(str(tmp_path), loader=loader)
    assert list(s.skills) == ["kick_left"] and s.sitstand is None
    s = load_set(str(tmp_path), walk="sprint.onnx", sitstand="got_a_talent.onnx", loader=loader)
    assert s.walk_file == "sprint.onnx" and s.sitstand.policy is nets["got_a_talent"]


class UprightIO(RobotIO):
    def __init__(self):
        self.t, self.torque, self.written, self.gravity = 0.0, False, [], np.array([0.0, 0.0, -1.0])
        self.gyro = np.zeros(3)
        self.torque_calls = []

    def read(self):
        return Sensors(positions=DEFAULT_POSITION.copy(), imu=ImuData(gravity=self.gravity.copy(), gyro=self.gyro.copy()),
                       imu_ready=True, t=self.t)

    def write(self, targets):
        self.written.append(targets.copy())

    def set_torque(self, on):
        self.torque = on
        self.torque_calls.append(on)

    def now(self):
        return self.t

    def wait_tick(self):
        self.t += self.control_dt


def running(tmp_path, tuning=None):
    loader, nets = make_set(tmp_path)
    s = load_set(str(tmp_path), loader=loader)
    io = UprightIO()
    r = Runner(io, Controller(s.walk, tuning or Tuning.official()), ramp_s=0.02, policies=s,
               head_offset=(0.0, 0.1, 0.0, 0.0))
    r.enable(); r.tick(); r.tick()
    assert r.state is State.HOLD and r.start()
    return r, nets, io


def drive(r, n, twist=(0.2, 0.0, 0.3), head=(0.0, 0.1, 0.0, 0.0)):
    for _ in range(n):
        r.set_command(Command(twist=twist, head=head), r.io.now())
        r.tick()


def test_kick_drives_its_network_for_its_window_with_a_zero_command(tmp_path):
    r, nets, _ = running(tmp_path)
    drive(r, 3)
    assert r.trigger("kick_left") is None
    assert r.trigger("kick_right") == "kick_left is still running"
    drive(r, 25)
    kick = nets["ball_kick_left"]
    assert len(kick.seen) == 25 and np.all(np.array(kick.seen) == 0.0)   # twist, head, body all zero
    assert r.mode == "kick_left"
    drive(r, 1)
    assert r.mode == "walk" and nets["velstand"].seen[-1][0] == pytest.approx(0.2)
    assert r.trigger("nope") == "no skill 'nope' loaded"


def test_skill_refused_outside_run(tmp_path):
    r, _, _ = running(tmp_path)
    r.stop()
    assert r.trigger("kick_left").startswith("policy not running")


def test_ground_pick_phase_encoding_and_scale(tmp_path):
    r, nets, io = running(tmp_path, Tuning(action_scale=0.9, head_lowpass=None, legs_lowpass=None))
    assert r.trigger("ground_pick") is None
    drive(r, 140)
    seen = np.array(nets["alpha_ground_pick"].seen)
    assert len(seen) == 140
    k = np.arange(140)
    np.testing.assert_allclose(seen[:, 0], np.cos(2 * math.pi * k * 0.02 / 4.0), atol=1e-6)
    np.testing.assert_allclose(seen[:, 1], np.sin(2 * math.pi * k * 0.02 / 4.0), atol=1e-6)
    assert np.all(seen[:, 2:] == 0.0)
    a = nets["alpha_ground_pick"].value
    leg = POLICY_JOINTS[0]
    assert io.written[-1][leg] == pytest.approx(DEFAULT_POSITION[leg] + 1.0 * a)   # its own scale 1.0
    drive(r, 1)
    assert r.mode == "walk"
    assert io.written[-1][leg] == pytest.approx(DEFAULT_POSITION[leg] + 0.9 * nets["velstand"].value)


def test_sit_settle_rise_then_walk(tmp_path):
    r, nets, _ = running(tmp_path)
    assert r.sit_toggle() is None and r.sit == "sitting"
    drive(r, 10)
    ss = np.array(nets["alpha_sitstand"].seen)
    assert np.all(ss[:, 0] == 1.0) and np.all(ss[:, 1:3] == 0.0)
    assert ss[-1][4] == pytest.approx(0.1)                       # head command stays live
    assert r.sit_toggle() == "still sitting down"
    assert r.trigger("kick_left") == "the robot is sitting: stand it up first"
    drive(r, 100)                                                # 2 s seat settle
    assert r.sit_toggle() is None and r.sit == "rising"
    drive(r, 50)
    assert nets["alpha_sitstand"].seen[-1][0] == 0.0 and r.mode == "rise"
    drive(r, 1)
    assert r.mode == "walk" and r.sit == "up"


def test_fall_does_not_limp_during_a_move(tmp_path):
    r, _, io = running(tmp_path)
    assert r.trigger("roulade") is None
    io.gravity = np.array([0.0, 0.0, 1.0])                       # upside down, by design
    drive(r, 50)
    assert r.state is State.RUN and r.fall_events == 0
    drive(r, 15)                                                 # handed back still over: grace
    assert r.state is State.RUN and r.fall_events == 0
    drive(r, 125)                                                # 2.5 s and still on its back
    assert r.state is State.LIMP and r.fall_events == 1


def test_limp_clears_moves(tmp_path):
    r, _, _ = running(tmp_path)
    r.sit_toggle()
    r.limp()
    assert r.sit == "up" and r.move is None and not r.busy



def test_keep_on_fall_starts_from_the_floor_and_gives_up_after_the_timeout(tmp_path):
    loader, nets = make_set(tmp_path)
    s = load_set(str(tmp_path), loader=loader)
    io = UprightIO()
    io.gravity = np.array([0.0, -1.0, 0.0])                      # lying on its side
    r = Runner(io, Controller(s.walk, Tuning.official()), ramp_s=0.02, policies=s)
    r.enable(); r.run_for(0.1)
    assert not r.start() and "keep-on-fall" in r.refusal          # default: refuse
    r.on_fall = "keep"
    drive(r, 300)                                                # lying 6 s in HOLD
    assert r.start()                                             # keep: the policy may get it up
    drive(r, 200)                                                # 4 s down: still trying
    assert r.state is State.RUN and r.falls_seen == 1
    io.gravity = np.array([0.0, 0.0, -1.0])                      # got up
    drive(r, 50)
    io.gravity = np.array([0.0, -1.0, 0.0])                      # falls again
    drive(r, 200)
    assert r.state is State.RUN and r.falls_seen == 2
    drive(r, 70)                                                 # > 5 s down
    assert r.state is State.LIMP and r.fall_events == 1 and "still down" in r.refusal


def test_getup_limp_fall_sequence(tmp_path):
    """robotd limp-fall: limp as the fall starts, land, pose to standing, hand back."""
    r, nets, io = running(tmp_path)
    r.on_fall = "getup"
    drive(r, 5)
    io.torque_calls.clear()
    n_walk = len(nets["velstand"].seen)
    # Tipping forward at 3 rad/s, 35 deg over: the predictor fires after its 60 ms debounce.
    tilt = math.radians(35)
    io.gravity = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    io.gyro = np.array([0.0, 3.0, 0.0])
    drive(r, 4)
    assert r.fall_seq and r.fall_seq["phase"] == "limp" and io.torque_calls == [False] and r.falls_seen == 1
    assert r.mode == "fall:limp"
    # Landed on its face, still: torque back on after 0.2 s of stillness, then a 0.6 s pose.
    io.gravity = np.array([1.0, 0.0, 0.0])
    io.gyro = np.zeros(3)
    drive(r, 12)
    assert r.fall_seq["phase"] == "pose" and io.torque_calls == [False, True]
    n_before = len(nets["velstand"].seen)
    drive(r, 25)
    assert len(nets["velstand"].seen) == n_before            # the policy is not stepped while posing
    drive(r, 8)
    assert r.fall_seq is None and r.mode == "walk" and r.state is State.RUN
    assert len(nets["velstand"].seen) > n_walk
    # Still down 5 s after the hand-back: give up and limp.
    drive(r, 260)
    assert r.state is State.LIMP and r.fall_events == 1 and "still down" in r.refusal


def test_getup_ignores_footfalls_moves_and_robots_already_down(tmp_path):
    r, _, io = running(tmp_path)
    r.on_fall = "getup"
    io.gyro = np.array([0.0, 3.0, 0.0])                      # an upright robot rocking: no fall
    tilt = math.radians(8)
    io.gravity = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    drive(r, 20)
    assert r.fall_seq is None
    assert r.trigger("roulade") is None                     # tipping over on purpose
    tilt = math.radians(40)
    io.gravity = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    drive(r, 20)
    assert r.fall_seq is None
