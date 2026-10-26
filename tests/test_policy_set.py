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
    assert s.skills["kick_left"].duration_s == 0.5 and s.skills["kick_left"].action_scale == 1.0  # tuned scale
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


def test_ground_pick_scripts_the_beak_then_hands_it_back(tmp_path):
    from duck_runtime.contract import MOUTH_CLOSED, MOUTH_INDEX
    from duck_runtime.runner import PICK_MOUTH_OPEN
    r, nets, io = running(tmp_path)
    r.mouth_target = math.radians(10)                      # whatever the user had set
    assert r.trigger("ground_pick") is None
    mouth = []
    for _ in range(140):
        drive(r, 1)
        mouth.append(io.written[-1][MOUTH_INDEX])
    mouth = np.degrees(mouth)
    k = lambda phase: int(round(phase * 4.0 / 0.02))       # tick of a phase (period 4 s)
    assert mouth[k(0.05)] == pytest.approx(np.degrees(MOUTH_CLOSED))
    assert mouth[k(0.35)] == pytest.approx(np.degrees(PICK_MOUTH_OPEN))   # open at the bottom
    assert mouth[k(0.50)] == pytest.approx(np.degrees(MOUTH_CLOSED))      # bitten, closed rising
    assert np.all(np.diff(mouth[:k(0.30)]) >= -1e-9)
    drive(r, 2)
    assert np.degrees(io.written[-1][MOUTH_INDEX]) == pytest.approx(10.0)   # back to the user's


# -- kissqy's HD1910 set: per-network low-pass and the set's own gait settings -----------------

def kissqy_set(tmp_path):
    from duck_runtime.fetch_set import kissqy_manifest
    return make_set(tmp_path, manifest=kissqy_manifest())


def test_kissqy_manifest_loads_with_its_deployment_settings(tmp_path):
    from duck_runtime.policy_set import apply_walk_tuning
    loader, nets = kissqy_set(tmp_path)
    s = load_set(str(tmp_path), loader=loader)
    assert s.walk is nets["kissqy_velstand"] and (s.walk_action_scale, s.walk_lowpass) == (1.0, (0.5, 0.7))
    assert s.sitstand.policy is nets["kissqy_sitstand"] and s.sitstand.lowpass is None   # stock sit/stand path
    assert list(s.skills) == ["ground_pick", "roulade", "kick_left", "kick_right"]
    assert all(sk.lowpass == (1.0, 1.0) and sk.action_scale == 1.0 for sk in s.skills.values())
    assert s.skills["ground_pick"].encoding == "phase"
    assert bind_keys(s.skills) == {"g": "ground_pick", "r": "roulade", "k": "kick_left", "l": "kick_right"}
    t = apply_walk_tuning(s, Tuning.official())
    assert (t.action_scale, t.head_lowpass, t.legs_lowpass) == (1.0, 0.5, 0.7)
    assert apply_walk_tuning(s, Tuning.official(), 0.8).action_scale == 0.8          # --action-scale wins
    # --policy replaces the walk: the manifest's walk settings no longer apply
    s2 = load_set(str(tmp_path), walk="kissqy_roulade.onnx", loader=loader)
    assert s2.walk_action_scale is None and s2.walk_lowpass is None


def test_official_manifest_keeps_the_global_tuning(tmp_path):
    from duck_runtime.policy_set import apply_walk_tuning
    loader, _ = make_set(tmp_path)
    s = load_set(str(tmp_path), loader=loader)
    assert s.walk_action_scale is None and all(sk.lowpass is None for sk in s.skills.values())
    t = apply_walk_tuning(s, Tuning.official())
    assert (t.action_scale, t.head_lowpass, t.legs_lowpass) == (0.9, 0.5, 0.7)


def test_unfiltered_skill_bypasses_the_global_lowpass(tmp_path):
    loader, nets = kissqy_set(tmp_path)
    from duck_runtime.policy_set import apply_walk_tuning
    s = load_set(str(tmp_path), loader=loader)
    io = UprightIO()
    r = Runner(io, Controller(s.walk, apply_walk_tuning(s, Tuning.official())), ramp_s=0.02, policies=s)
    r.enable(); r.tick(); r.tick()
    assert r.start()
    drive(r, 5)
    leg = POLICY_JOINTS[3]
    walk_target = io.written[-1][leg]
    assert r.trigger("kick_left") is None
    drive(r, 1)
    a = nets["kissqy_kick_left"].value
    assert io.written[-1][leg] == pytest.approx(DEFAULT_POSITION[leg] + a)          # no 0.7 blend
    assert io.written[-1][leg] != pytest.approx(0.7 * (DEFAULT_POSITION[leg] + a) + 0.3 * walk_target)
    drive(r, 25)                                                                    # back to the gait
    assert r.mode == "walk"
    w = nets["kissqy_velstand"].value
    prev = io.written[-2][leg]
    assert io.written[-1][leg] == pytest.approx(0.7 * (DEFAULT_POSITION[leg] + 1.0 * w) + 0.3 * prev)


def test_lowpass_out_of_range_is_refused(tmp_path):
    m = json.loads(json.dumps(OFFICIAL_MANIFEST))
    m["policies"][-1]["lowpass"] = {"head": 1.0, "legs": 0.0}
    loader, _ = make_set(tmp_path, manifest=m)
    with pytest.raises(ValueError, match="lowpass"):
        load_set(str(tmp_path), loader=loader)


def test_load_sets_two_dirs_first_active_policy_override_on_the_first_only(tmp_path, monkeypatch):
    import argparse
    import duck_runtime.policy_set as ps
    from duck_runtime.fetch_set import kissqy_manifest
    from duck_runtime.run import add_runner_args, load_sets
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies_hd1910").mkdir()
    loader, nets = make_set(tmp_path / "policies")
    loader2, nets2 = make_set(tmp_path / "policies_hd1910", manifest=kissqy_manifest())
    real_load_set = ps.load_set
    monkeypatch.setattr(ps, "load_set", lambda d, **kw: real_load_set(
        d, loader=lambda path: (loader2 if "hd1910" in path else loader)(path), **kw))
    ap = argparse.ArgumentParser()
    add_runner_args(ap)
    args = ap.parse_args(["--set", str(tmp_path / "policies_hd1910"), "--set", str(tmp_path / "policies")])
    bank = load_sets(args, ap, Tuning.official())
    assert [s.label for s in bank.sets] == ["policies_hd1910 (hd1910)", "policies (xl330)"]
    assert bank.active.label.startswith("policies_hd1910") and bank.active.tuning.action_scale == 1.0
    assert bank.sets[1].tuning.action_scale == 0.9                # official: global tuning untouched
    assert bank.sets[0].tuning is not bank.sets[1].tuning
    r = Runner(UprightIO(), Controller(bank.active.pset.walk, bank.active.tuning), ramp_s=0.02,
               policies=bank.active.pset)
    r.enable(); r.tick(); r.tick()
    assert r.start() and bank.switch(r, 1) is not None           # running: refused
    r.stop()
    assert bank.switch(r, 1) is None and r.controller.policy is nets["velstand"] and bank.index == 1
    assert r.controller.tuning.action_scale == 0.9
    args = ap.parse_args(["--set", str(tmp_path / "policies"), "--set", str(tmp_path / "policies_hd1910"),
                          "--action-scale", "0.8"])
    bank = load_sets(args, ap, Tuning.official())
    assert [s.tuning.action_scale for s in bank.sets] == [0.8, 0.8]   # an explicit override wins everywhere
