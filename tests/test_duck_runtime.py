"""duck_runtime: the policy contract, IMU decode, controller semantics, Feetech codec, sim loop.

CPU only. The contract tests pin duck_runtime against microduck_rl's own training-side
definitions (MJCF actuator order, infer_policy's HOME), so a drift on either side fails here
instead of on the robot.
"""

import importlib.util
import math
import struct
from pathlib import Path

import mujoco
import numpy as np
import pytest

from duck_runtime import contract as C
from duck_runtime.calibration import Calibration, JointCal
from duck_runtime.controller import Command, Controller, Tuning
from duck_runtime.imu import DEFAULT_MOUNT, SflpDecoder, encode_block, quat_mul, rotate_inverse
from duck_runtime.io import Sensors

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"


# -- contract ------------------------------------------------------------------------------

def test_policy_joint_order_matches_training_actuators():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    actuated = [m.actuator(i).name for i in range(m.nu)]
    assert [C.JOINT_NAMES[j] for j in C.POLICY_JOINTS] == actuated


def test_home_matches_training_default_pose():
    spec = importlib.util.spec_from_file_location("ip", REPO / "scripts/infer_policy.py")
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    np.testing.assert_allclose(C.policy_joints(C.DEFAULT_POSITION), ip.DEFAULT_POSE, atol=1e-6)


def test_observation_layout():
    q = np.arange(15) * 0.01 + C.DEFAULT_POSITION
    v = np.arange(15) * 0.1
    cmd = C.command_block((1, 2, 3), (4, 5, 6, 7), body_z=8, body_roll=9, body_pitch=10)
    obs = C.build_observation([.1, .2, .3], [0, 0, -1], q, v, np.full(14, 7.0), cmd)
    assert obs.shape == (61,)
    np.testing.assert_allclose(obs[0:6], [.1, .2, .3, 0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[6:20], C.policy_joints(np.arange(15) * 0.01), atol=1e-6)
    assert 9 * 0.1 not in obs[20:34]  # mouth velocity excluded
    np.testing.assert_allclose(obs[34:48], 7.0)
    # twist, head, body x=0, y=0, z, roll, pitch, yaw=0 (official obs.rs order)
    np.testing.assert_allclose(obs[48:61], [1, 2, 3, 4, 5, 6, 7, 0, 0, 8, 9, 10, 0])


def test_scatter_leaves_mouth_slot():
    a = np.arange(1, 15, dtype=float)
    out = C.scatter_action(a)
    assert out[C.MOUTH_INDEX] == 0.0
    np.testing.assert_array_equal(C.policy_joints(out), a)


# -- IMU -----------------------------------------------------------------------------------

def _rand_quat(rng, min_w=0.3):
    """Random orientation with w >= min_w. The chip ships xyz as IEEE half and the
    decoder rebuilds w = sqrt(1-|xyz|^2): at small w that amplifies the ~1e-3 half
    rounding (official decoder alike), so near-180-degree attitudes are excluded."""
    while True:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] > 0 else -q
        if q[0] >= min_w:
            return q


@pytest.mark.parametrize("mount", [DEFAULT_MOUNT, (1.0, 0.0, 0.0, 0.0)])
def test_sflp_roundtrip_through_mount(mount):
    rng = np.random.default_rng(0)
    done = 0
    while done < 20:
        q_trunk = _rand_quat(rng)
        if abs(quat_mul(q_trunk, mount)[0]) < 0.3:  # the bound applies to the SENSOR quaternion
            continue
        done += 1
        gyro_trunk = rng.uniform(-3, 3, 3)
        block = encode_block(rotate_inverse(mount, gyro_trunk), quat_mul(q_trunk, mount))
        dec = SflpDecoder(mount)
        for _ in range(3):  # fill the median history
            out = dec.decode(block)
        g_true = rotate_inverse(q_trunk, np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(out.gravity, g_true, atol=3e-3)  # half-precision quat
        np.testing.assert_allclose(out.gyro, gyro_trunk, atol=2e-3)


def test_sflp_zero_quaternion_keeps_last_good():
    dec = SflpDecoder()
    out = dec.decode(struct.pack("<hhh", 0, 0, 0) + b"\x00" * 6)
    np.testing.assert_allclose(out.gravity, [0, 0, -1])
    assert not dec.ready


def test_imu_mount_from_two_poses():
    from duck_runtime.tools import mount_from_two_poses
    m_true = np.array(DEFAULT_MOUNT)
    up = rotate_inverse(m_true, np.array([0.0, 0.0, -1.0]))                         # sensor frame
    th = math.radians(40)
    nose_down = rotate_inverse(m_true, np.array([math.sin(th), 0.0, -math.cos(th)]))
    q = mount_from_two_poses(up, nose_down)
    assert abs(abs(np.dot(q, m_true)) - 1.0) < 1e-9


# -- controller ------------------------------------------------------------------------------

class _ConstPolicy:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def __call__(self, obs):
        return self.a.copy()


def test_controller_scale_lowpass_and_raw_last_action():
    a = np.full(14, 0.2, dtype=np.float32)
    c = Controller(_ConstPolicy(a), Tuning.official())
    s = Sensors(positions=C.DEFAULT_POSITION.copy())
    t1 = c.step(s, Command())
    np.testing.assert_allclose(c.last_action, a)                       # raw, not scaled
    np.testing.assert_allclose(C.policy_joints(t1), C.policy_joints(C.DEFAULT_POSITION) + 0.9 * 0.2)
    assert t1[C.MOUTH_INDEX] == pytest.approx(C.MOUTH_CLOSED)
    c.policy = _ConstPolicy(np.zeros(14))
    t2 = c.step(s, Command())
    # legs: 0.7*home + 0.3*prev ; head: 0.5*home + 0.5*prev
    assert t2[0] - C.DEFAULT_POSITION[0] == pytest.approx(0.3 * 0.18)
    assert t2[5] - C.DEFAULT_POSITION[5] == pytest.approx(0.5 * 0.18)
    obs = c.observe(s, Command())
    np.testing.assert_allclose(obs[34:48], 0.0)


def test_velocity_lags_one_tick():
    c = Controller(_ConstPolicy(np.zeros(14)), Tuning.official())
    o1 = c.observe(Sensors(velocities=np.full(15, 1.0)), Command())
    o2 = c.observe(Sensors(velocities=np.full(15, 2.0)), Command())
    assert np.all(o1[20:34] == 1.0) and np.all(o2[20:34] == 1.0)


# -- calibration / Feetech codec ---------------------------------------------------------------

def test_joint_cal_roundtrip_and_clamp():
    j = JointCal("left_knee", 23, zero_raw=1500, direction=-1, min_rad=-1.0, max_rad=1.0)
    for rad in (-0.9, 0.0, 0.5):
        assert j.to_rad(j.to_raw(rad)) == pytest.approx(rad, abs=2 * math.pi / 4096)
    assert j.to_raw(5.0) == j.to_raw(1.0)


def test_template_validates_and_requires_measurement():
    cal = Calibration.template()
    cal.validate()
    with pytest.raises(ValueError):
        cal.validate(require_calibrated=True)


def test_feetech_packet_and_signmag():
    from duck_runtime.feetech import _packet, _signmag, _signmag_word
    assert _packet(1, 0x01) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])  # ping ID 1
    assert _signmag(0x8005) == -5 and _signmag(0x0005) == 5
    assert _signmag_word(-5) == 0x8005


def test_feetech_feedback_decode():
    from duck_runtime.feetech import FeetechIO, SPEED_RAD_S_PER_COUNT
    io = FeetechIO.__new__(FeetechIO)
    io.cal = Calibration.template()
    blocks = {}
    for k, j in enumerate(io.cal.joints):
        raw = j.to_raw(0.1 * (k % 3))
        blocks[j.id] = struct.pack("<HHH", raw, 0x8000 | 10, 0) + bytes([75, 30]) + b"\x00" * 7
    pos, vel, volt, temp = io._decode(blocks)
    np.testing.assert_allclose(pos, [0.1 * (k % 3) for k in range(15)], atol=2e-3)
    np.testing.assert_allclose(vel, [-10 * SPEED_RAD_S_PER_COUNT * j.direction for j in io.cal.joints])
    assert volt.min() == pytest.approx(7.5) and temp.max() == 30


# -- the whole loop, simulated ---------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_runner():
    pytest.importorskip("bam")
    from duck_runtime.runner import Runner
    from duck_runtime.sim_io import SimIO

    io = SimIO(mass_profile="ours_813g", servo="hd1910")
    return Runner(io, Controller(_ConstPolicy(np.zeros(14)), Tuning.official()), ramp_s=0.5)


def test_runner_never_moves_without_a_request(sim_runner):
    from duck_runtime.runner import State
    r = sim_runner
    r.io.reset()
    r.state = State.LIMP
    r.run_for(0.2)
    assert r.state is State.LIMP and not r.io.torque


def test_runner_ramp_hold_run_and_deadman(sim_runner):
    from duck_runtime.runner import State
    r = sim_runner
    r.io.reset()
    r.state = State.LIMP
    r.enable()
    r.run_for(0.8)
    assert r.state is State.HOLD
    r.start()
    r.set_command(Command(twist=(0.3, 0, 0)), r.io.now())
    r.tick()
    assert r.controller.last_obs[48] == pytest.approx(0.3)
    r.run_for(0.6)  # no fresh command for > 0.5 s
    assert r.controller.last_obs[48] == 0.0


def test_runner_limps_on_fall(sim_runner):
    from duck_runtime.runner import State
    r = sim_runner
    r.io.reset()
    r.state = State.LIMP
    r.enable(); r.run_for(0.8); r.start()
    r.io.push(1.5, 0.0)
    r.run_for(2.0)
    assert r.fall_events >= 1 and r.state is State.LIMP and not r.io.torque


def test_runner_refuses_to_start_when_lying(sim_runner):
    from duck_runtime.runner import State
    r = sim_runner
    r.io.reset()
    r.state = State.LIMP
    r.run_for(3.0)          # unpowered: folds onto the floor
    r.enable(); r.run_for(0.8)
    assert r.state is State.HOLD
    assert r.start() is False and r.refusal == "robot is not upright"
    assert r.state is State.HOLD


# -- ESP32 IMU bridge ------------------------------------------------------------------------

def _block_for(q_trunk, gyro_trunk, mount=DEFAULT_MOUNT):
    return encode_block(rotate_inverse(mount, gyro_trunk), quat_mul(q_trunk, mount))


def test_frame_parser_resyncs_and_checks_crc():
    from duck_runtime.esp32_link import FrameParser, make_frame
    blk = bytes(range(12))
    good = make_frame(7, 3, blk)
    bad = bytearray(make_frame(8, 3, blk)); bad[10] ^= 0xFF
    p = FrameParser()
    out = p.feed(b"\x00\xa5\x13" + bytes(bad) + good[:5]) + p.feed(good[5:] + make_frame(9, 1, blk))
    assert [(s, f) for s, f, _ in out] == [(7, 3), (9, 1)] and out[0][2] == blk
    assert p.bad_crc == 1


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _NullSerial:
    in_waiting = 0

    def read(self, n):
        import time as _t
        _t.sleep(0.001)
        return b""

    def close(self):
        pass


def test_esp32_imu_readiness_and_staleness():
    from duck_runtime.esp32_link import Esp32Imu, make_frame
    clk = _FakeClock()
    imu = Esp32Imu(serial_obj=_NullSerial(), clock=clk)
    q = np.array([math.cos(0.1), 0.0, math.sin(0.1), 0.0])  # pitched ~11.5 deg
    blk = _block_for(q, np.array([0.1, -0.2, 0.3]))
    _, ready = imu.read()
    assert not ready
    for k in range(30):
        clk.t = k * 0.005
        imu.ingest(make_frame(k, 3, blk))
        data, ready = imu.read()
    assert ready
    np.testing.assert_allclose(data.gravity, rotate_inverse(q, np.array([0, 0, -1.0])), atol=3e-3)
    np.testing.assert_allclose(data.gyro, [0.1, -0.2, 0.3], atol=2e-3)
    clk.t += 0.06                      # link silent for 60 ms
    assert imu.read()[1] is False
    imu.ingest(make_frame(99, 2, blk))  # frames resume but no fresh quaternion for > 100 ms
    clk.t += 0.06
    imu.ingest(make_frame(100, 2, blk))
    assert imu.read()[1] is False
    imu.ingest(make_frame(101, 3 | 4, blk))  # sensor error flag
    assert imu.read()[1] is False
    imu.close()


def test_esp32_imu_over_a_real_pty():
    """The reader thread + pyserial against a pseudo-terminal streaming frames at 200 Hz."""
    import os, pty, threading, time as _t
    serial = pytest.importorskip("serial")
    from duck_runtime.esp32_link import Esp32Imu, make_frame
    master, slave = pty.openpty()
    imu = Esp32Imu(os.ttyname(slave), baud=921600)
    stop = threading.Event()
    blk = _block_for(np.array([1.0, 0, 0, 0]), np.zeros(3))

    def writer():
        k = 0
        while not stop.is_set():
            os.write(master, make_frame(k, 3, blk)); k += 1
            _t.sleep(0.005)

    th = threading.Thread(target=writer, daemon=True); th.start()
    try:
        _t.sleep(0.4)
        for _ in range(30):
            data, ready = imu.read()
            _t.sleep(0.005)
        assert imu.frames > 40 and ready
        np.testing.assert_allclose(data.gravity, [0, 0, -1], atol=3e-3)
    finally:
        stop.set(); th.join(); imu.close(); os.close(master); os.close(slave)


# -- Feetech bus against a simulated HD1910 bus on a pty --------------------------------------

class _FakeFeetechBus:
    """Answers like 15 HD1910s + an FT6 IMU: PING / READ / WRITE / SYNC_READ / SYNC_WRITE
    on a flat 256-byte register file per ID. Optionally echoes what the host sends."""

    def __init__(self, fd, ids, echo=False, imu_block=None):
        import threading
        self.fd, self.echo = fd, echo
        self.regs = {i: bytearray(256) for i in ids}
        for i in ids:
            self.regs[i][56:58] = (2048).to_bytes(2, "little")
            self.regs[i][62] = 76  # 7.6 V
        if imu_block is not None:
            self.regs[200] = bytearray(256)
            self.regs[200][56:68] = imu_block
        self.buf = b""
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _reply(self, id_, data=b""):
        body = bytes([id_, len(data) + 2, 0]) + data
        import os
        os.write(self.fd, b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF]))

    def _run(self):
        import os, select
        while not self.stop.is_set():
            if not select.select([self.fd], [], [], 0.01)[0]:
                continue
            chunk = os.read(self.fd, 4096)
            self.buf += chunk
            while len(self.buf) >= 6:
                i = self.buf.find(b"\xff\xff")
                if i < 0:
                    self.buf = b""; break
                self.buf = self.buf[i:]
                n = self.buf[3] + 4
                if len(self.buf) < n:
                    break
                pkt, self.buf = self.buf[:n], self.buf[n:]
                if self.echo:
                    os.write(self.fd, pkt)
                id_, ins, p = pkt[2], pkt[4], pkt[5:-1]
                if ins == 0x01 and id_ in self.regs:
                    self._reply(id_)
                elif ins == 0x02 and id_ in self.regs:
                    self._reply(id_, bytes(self.regs[id_][p[0]:p[0] + p[1]]))
                elif ins == 0x03 and id_ in self.regs:
                    self.regs[id_][p[0]:p[0] + len(p) - 1] = p[1:]
                    self._reply(id_)
                elif ins == 0x82:
                    addr, ln = p[0], p[1]
                    for rid in p[2:]:
                        if rid in self.regs:
                            self._reply(rid, bytes(self.regs[rid][addr:addr + ln]))
                elif ins == 0x83:
                    addr, w = p[0], p[1]
                    rows = p[2:]
                    for k in range(0, len(rows), w + 1):
                        rid = rows[k]
                        if rid in self.regs:
                            self.regs[rid][addr:addr + w] = rows[k + 1:k + 1 + w]
                            if addr == 42:  # goal -> present position (perfect servo)
                                self.regs[rid][56:58] = rows[k + 1:k + 3]

    def close(self):
        self.stop.set(); self.th.join()


def _eventually(pred, timeout=0.5):
    """Sync writes get no status reply, so the host returns before the fake bus applied them."""
    import time as _t
    end = _t.monotonic() + timeout
    while _t.monotonic() < end:
        if pred():
            return True
        _t.sleep(0.005)
    return pred()


@pytest.mark.parametrize("echo", [False, True])
def test_feetech_io_against_simulated_bus(echo):
    import os, pty
    pytest.importorskip("serial")
    from duck_runtime.feetech import FeetechIO, REG_RAM_P, REG_TORQUE
    master, slave = pty.openpty()
    import tty as _tty
    _tty.setraw(master); _tty.setraw(slave)
    q = np.array([math.cos(0.05), math.sin(0.05), 0.0, 0.0])
    fake = _FakeFeetechBus(master, list(C.JOINT_IDS), echo=echo, imu_block=_block_for(q, np.zeros(3)))
    cal = Calibration.template()
    io = FeetechIO(os.ttyname(slave), cal)
    try:
        s = io.read()
        assert io.bus.echo is echo
        np.testing.assert_allclose(s.positions, [0.0] * 15, atol=1e-9)  # raw 2048 = zero_raw
        assert io.voltages.min() == pytest.approx(7.6)
        for _ in range(2):  # 3-sample median: the first two reads still hold the default
            s = io.read()
        np.testing.assert_allclose(s.imu.gravity, rotate_inverse(q, [0, 0, -1.0]), atol=3e-3)
        io.set_torque(True)
        assert _eventually(lambda: all(fake.regs[i][REG_RAM_P] == 5 and fake.regs[i][51] == 0
                                       and fake.regs[i][REG_TORQUE] == 1 for i in C.JOINT_IDS))
        assert all(int.from_bytes(fake.regs[i][48:50], "little") == 1000 for i in C.JOINT_IDS)
        target = C.DEFAULT_POSITION.copy()
        io.write(target)
        _eventually(lambda: fake.regs[C.JOINT_IDS[2]][56:58] != (2048).to_bytes(2, "little"))
        s = io.read()
        expect = [j.to_rad(j.to_raw(t)) for j, t in zip(cal.joints, target)]
        np.testing.assert_allclose(s.positions, expect, atol=1e-9)
        io.set_torque(False)
        assert _eventually(lambda: all(fake.regs[i][REG_TORQUE] == 0 for i in C.JOINT_IDS))
    finally:
        io.close(); fake.close(); os.close(master)
