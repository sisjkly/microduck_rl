"""This robot's I/O: the ESP32-S3 MDK2 bridge (USB Dynamixel 2.0 -> HD1910) and the Pi I2C
LSM6DSV16X. The bridge is emulated on a real pty by a Python port of the subset of
bridge_runtime.cpp that BridgeIO touches (status/diag/NPD1 registers, sync read/write,
torque/fault rules, the center/direction goal mapping); the IMU by a register-level fake.
"""

import math
import os
import pty
import struct
import threading
import time

import numpy as np
import pytest

from duck_runtime import bridge as B
from duck_runtime import contract as C
from duck_runtime.calibration import BRIDGE_IDS, Calibration
from duck_runtime.imu import IDENTITY_MOUNT, ImuData, quat_mul, rotate_inverse
from duck_runtime import pi_imu as P


# -- framing ---------------------------------------------------------------------------------

def test_packet_stuffing_crc_and_parser_resync():
    # A status whose payload contains FF FF FD must be stuffed and come back intact.
    payload = b"\x55\x00" + b"\xff\xff\xfd\x01\x02"
    body = B.stuff(payload)
    assert body.count(b"\xfd") == payload.count(b"\xfd") + 1
    frame = b"\xff\xff\xfd\x00\x07" + struct.pack("<H", len(body) + 2) + body
    frame += struct.pack("<H", B.crc16(frame))
    corrupt = bytearray(frame)
    corrupt[-1] ^= 0xFF
    p = B.StatusParser()
    out = p.feed(b"\x00\x13" + bytes(corrupt) + frame[:5])
    assert out == [] and p.bad == 1
    assert p.feed(frame[5:]) == [(7, 0, b"\xff\xff\xfd\x01\x02")]
    # Known vector from the ROBOTIS Protocol 2.0 docs: ping ID 1.
    assert B.packet(1, B.INS_PING) == bytes.fromhex("FFFFFD0001030001194E")


# -- emulated bridge firmware ----------------------------------------------------------------

def _parse_instructions(buf: bytes):
    out = []
    while True:
        k = buf.find(b"\xff\xff\xfd\x00")
        if k < 0 or len(buf) - k < 7:
            return out, buf[k:] if k >= 0 else b""
        buf = buf[k:]
        n = 7 + struct.unpack_from("<H", buf, 5)[0]
        if len(buf) < n:
            return out, buf
        frame, buf = buf[:n], buf[n:]
        assert B.crc16(frame[:-2]) == struct.unpack_from("<H", frame, n - 2)[0]
        body = B.unstuff(frame[7:-2])
        out.append((frame[4], body[0], body[1:]))


class FakeBridge:
    """bridge_runtime.cpp in miniature: serves IDs 1..n (14 stock, 15 with add_mouth.py)."""

    def __init__(self, master, n_servos=14):
        self.fd = master
        self.ids = tuple(range(1, n_servos + 1))
        self.ALL = (1 << n_servos) - 1
        self.raw = {i: 2048 + 10 * i for i in self.ids}
        self.speed = {i: 0 for i in self.ids}
        self.pd = {i: [32, 40] for i in self.ids}
        self.center = {i: 2048 for i in self.ids}
        self.dir = {i: -1 for i in self.ids}
        self.goal_ft = {}
        self.torque = 0
        self.fault = 0
        self.mode = 4
        self.speed_limit = 0
        self.alive = True
        self.mute = False            # an ESP32 that stopped answering (rebooting, USB gone)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _status(self, id_, err, data=b""):
        body = B.stuff(bytes([0x55, err]) + data)
        f = b"\xff\xff\xfd\x00" + bytes([id_]) + struct.pack("<H", len(body) + 2) + body
        os.write(self.fd, f + struct.pack("<H", B.crc16(f)))

    def _reg(self, i, addr, n):
        if addr == 256 and n == 64:
            b = bytearray(64)
            b[:4] = b"MDK2"
            struct.pack_into("<HHHHH", b, 4, 2, self.ALL, self.ALL, self.torque, self.fault)
            struct.pack_into("<H", b, 48, self.ALL)
            return bytes(b)
        if addr == 320 and n == 32:
            b = bytearray(32)
            dx = 2048 + self.dir[i] * (self.raw[i] - self.center[i])
            struct.pack_into("<iihhH", b, 0, self.raw[i], dx, self.speed[i], 120, 75)
            b[14], b[15] = 30, 0
            struct.pack_into("<ibBB", b, 20, self.center[i], self.dir[i], self.mode, 1)
            b[27] = 1 | 2 | (4 if self.torque >> (i - 1) & 1 else 0) | 8
            struct.pack_into("<HH", b, 28, 0, 4095)
            return bytes(b)
        if addr == 360 and n == 16:
            return b"NPD1" + bytes([1, 0] + self.pd[i]) + bytes(8)
        if 1024 <= addr and addr + n <= 1280:   # read-only native register window
            mem = bytearray(256)
            mem[50:52] = bytes(self.pd[i])
            mem[46:48] = struct.pack("<H", self.speed_limit)
            return bytes(mem[addr - 1024:addr - 1024 + n])
        return None

    def _write(self, addr, width, rows):
        rows = [(i, v) for i, v in rows if i in self.raw]     # dxIndex < 0: ignored
        if addr == 64:
            vals = {i: v[0] for i, v in rows}
            if any(vals.values()):
                if self.fault:
                    return
                if self.mode != 4:
                    self.fault = 5
                    self.torque = 0
                    return
            for i, v in vals.items():
                self.torque = self.torque | (1 << (i - 1)) if v else self.torque & ~(1 << (i - 1))
                if v and i in self.goal_ft:
                    self.raw[i] = self.goal_ft[i]
            if not any(vals.values()) and len(vals) == len(self.ids):
                self.fault = 0
        elif addr == 116:
            for i, v in rows:
                dx, = struct.unpack("<i", v)
                ft = self.center[i] + self.dir[i] * (dx - 2048)
                if not 0 <= ft <= 4095:
                    self.fault, self.torque = 4, 0
                    return
                self.goal_ft[i] = ft
                if self.torque >> (i - 1) & 1:
                    self.raw[i] = ft  # perfect servo

    def _run(self):
        buf = b""
        while self.alive:
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                return
            pkts, buf = _parse_instructions(buf + chunk)
            if self.mute:
                continue
            for id_, ins, p in pkts:
                if ins == B.INS_PING:
                    if id_ in self.raw:
                        self._status(id_, 0, b"\xb0\x04\x02")
                elif ins == B.INS_READ:
                    addr, n = struct.unpack("<HH", p)
                    d = self._reg(id_, addr, n)
                    self._status(id_, 0x04 if d is None else 0, d or b"")
                elif ins == B.INS_SYNC_READ:
                    addr, n = struct.unpack_from("<HH", p)
                    for i in p[4:]:
                        if i in self.raw:
                            self._status(i, 0, self._reg(i, addr, n))
                elif ins == B.INS_SYNC_WRITE:
                    addr, width = struct.unpack_from("<HH", p)
                    rows = [(p[k], p[k + 1:k + 1 + width]) for k in range(4, len(p), width + 1)]
                    self._write(addr, width, rows)
                elif ins == B.INS_NPD:
                    ok = p[:4] == b"NPD1" and not self.torque and not self.fault
                    if ok:
                        self.pd[id_] = [p[4], p[5]]
                    self._status(id_, 0 if ok else 0x80)


class StaticImu:
    def __init__(self):
        self.closed = False

    def read(self):
        return ImuData(), True

    def latest_block(self):
        return None

    def close(self):
        self.closed = True


def _calibrated():
    cal = Calibration.template_bridge()
    rng = np.random.default_rng(3)
    for j in cal.joints:
        if j.id:
            j.zero_raw = int(2048 + rng.integers(-200, 200))
            j.direction = int(rng.choice([-1, 1]))
            j.calibrated = True
    cal.imu_mount_verified = True
    return cal


@pytest.fixture
def rig():
    master, slave = pty.openpty()
    fake = FakeBridge(master)
    # Non-default firmware maps on two joints: BridgeIO must invert whatever is configured.
    fake.center[3], fake.dir[3] = 2100, 1
    fake.center[12], fake.dir[12] = 1900, -1
    cal = _calibrated()
    io = B.BridgeIO(os.ttyname(slave), cal, StaticImu())
    yield fake, io, cal
    fake.alive = False
    try:
        io.close()
    except OSError:
        pass
    os.close(slave)


def _eventually(pred, timeout=0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_bridge_template_matches_robot_wiring():
    cal = Calibration.template_bridge()
    cal.validate()
    by_name = {j.name: j.id for j in cal.joints}
    # ~/microduck/src/constants.py MOTOR_TO_ID on the robot.
    assert by_name == {"right_hip_yaw": 5, "right_hip_roll": 4, "right_hip_pitch": 3, "right_knee": 2,
                       "right_ankle": 1, "left_hip_yaw": 10, "left_hip_roll": 9, "left_hip_pitch": 8,
                       "left_knee": 7, "left_ankle": 6, "neck_pitch": 12, "head_pitch": 11,
                       "head_yaw": 13, "head_roll": 14, "mouth": 0}
    with pytest.raises(ValueError):
        cal.validate(require_calibrated=True)
    _calibrated().validate(require_calibrated=True)  # mouth (id 0) needs no measurement


def test_bridge_read_converts_with_python_calibration(rig):
    fake, io, cal = rig
    fake.speed[8] = -40
    s = io.read()
    for k, j in enumerate(cal.joints):
        if j.id:
            assert s.positions[k] == pytest.approx((fake.raw[j.id] - j.zero_raw) * 2 * math.pi / 4096 * j.direction)
    assert s.positions[C.MOUTH_INDEX] == C.MOUTH_CLOSED
    k8 = [j.id for j in cal.joints].index(8)
    assert s.velocities[k8] == pytest.approx(-40 * 50 * 2 * math.pi / 4096 * cal.joints[k8].direction)
    assert s.imu_ready


def test_bridge_torque_on_sets_trained_gains_holds_pose_then_tracks(rig):
    fake, io, cal = rig
    before = dict(fake.raw)
    io.set_torque(True)
    assert all(fake.pd[i] == [5, 0] for i in fake.ids)
    assert fake.torque == fake.ALL
    assert fake.raw == before  # goal = present position: no jump at torque-on
    target = C.DEFAULT_POSITION.copy()
    io.write(target)
    assert _eventually(lambda: all(fake.raw[j.id] == j.to_raw(t) for j, t in zip(cal.joints, target) if j.id))
    s = io.read()
    for k, j in enumerate(cal.joints):
        if j.id:
            assert s.positions[k] == pytest.approx(target[k], abs=2 * math.pi / 4096)


def test_bridge_fault_refuses_torque_and_limp_clears_it(rig):
    fake, io, _ = rig
    fake.fault = 1
    with pytest.raises(B.BridgeError, match="host timeout"):
        io.set_torque(True)
    assert fake.torque == 0
    io.set_torque(False)
    assert _eventually(lambda: fake.fault == 0)
    io.set_torque(True)
    assert fake.torque == fake.ALL


def test_bridge_close_restores_original_gains(rig):
    fake, io, _ = rig
    io.set_torque(True)
    io.close()
    assert fake.torque == 0
    assert all(fake.pd[i] == [32, 40] for i in fake.ids)


def test_bridge_port_is_exclusive(rig):
    _, io, _ = rig
    path = os.ttyname(io.bridge.port.fd)
    with pytest.raises(B.BridgeError, match="in use"):
        B.Port(path)


def test_runner_on_bridge_ramps_to_home():
    from duck_runtime.controller import Controller, Tuning
    from duck_runtime.runner import Runner, State
    from duck_runtime.safety import Safety

    class _Zero:
        def __call__(self, obs):
            return np.zeros(14)

    master, slave = pty.openpty()
    fake = FakeBridge(master)
    cal = _calibrated()
    io = B.BridgeIO(os.ttyname(slave), cal, StaticImu(), control_dt=0.002)
    try:
        r = Runner(io, Controller(_Zero(), Tuning.official()), Safety(cal.limits()), ramp_s=0.05)
        r.enable()
        for _ in range(60):
            r.tick()
        assert r.state is State.HOLD
        assert all(abs(fake.raw[j.id] - j.to_raw(h)) <= 1 for j, h in zip(cal.joints, C.DEFAULT_POSITION) if j.id)
    finally:
        fake.alive = False
        io.close()
        os.close(slave)


# -- Pi I2C LSM6DSV16X -------------------------------------------------------------------------

class FakeLsm6dsv:
    """Register-level fake: main + embedded banks, SW_POR, FIFO of 7-byte tagged words."""

    def __init__(self):
        self.addr = 0x6A
        self.main = {P.WHO_AM_I: P.CHIP_ID}
        self.emb = {P.SFLP_ODR: 0x4B}  # datasheet default: reserved bits set
        self.fifo: list[bytes] = []
        self.gyro = b"\x00" * 6
        self.fail = False

    def _bank(self, reg=None):
        if reg == P.FUNC_CFG_ACCESS:  # visible from both banks on the real chip
            return self.main
        return self.emb if self.main.get(P.FUNC_CFG_ACCESS, 0) & P.EMB_ACCESS else self.main

    def read(self, reg, n):
        if self.fail:
            raise OSError("NACK")
        bank = self._bank(reg)
        if bank is self.main and reg == P.FIFO_STATUS1:
            return bytes([len(self.fifo) & 0xFF, (len(self.fifo) >> 8) & 1])
        if bank is self.main and reg == P.FIFO_DATA_OUT_TAG:
            return self.fifo.pop(0)
        if bank is self.main and reg == P.OUTX_L_G:
            return self.gyro
        return bytes([bank.get(reg, 0)])

    def write(self, reg, value):
        if self.fail:
            raise OSError("NACK")
        bank = self._bank(reg)
        if bank is self.main and reg == P.FUNC_CFG_ACCESS and value & P.SW_POR:
            self.main = {P.WHO_AM_I: P.CHIP_ID}
            self.emb = {P.SFLP_ODR: 0x4B}
            self.fifo.clear()
            return
        if bank is self.main and reg == P.FIFO_CTRL4 and value & 7 == 0:
            self.fifo.clear()
        bank[reg] = value

    def push_quat(self, q_sensor):
        q = np.asarray(q_sensor, float)
        q = -q if q[0] < 0 else q
        self.fifo.append(bytes([(P.TAG_GAME_RV << 3) | 0x2]) + struct.pack("<eee", *q[1:]))

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_pi_imu_configures_like_the_st_driver():
    dev = FakeLsm6dsv()
    P.PiImu(dev=dev, mount=IDENTITY_MOUNT)
    m, e = dev.main, dev.emb
    assert m[P.CTRL1] & 0x7F == 0x06 and m[P.CTRL2] & 0x7F == 0x06       # 120 Hz high-performance
    assert m[P.CTRL6] & 0x0F == 0x02 and m[P.CTRL8] & 0x03 == 0x01       # 500 dps, 4 g
    assert m[P.CTRL3] & 0x44 == 0x44 and m[P.FIFO_CTRL4] & 7 == 6       # BDU|IF_INC, stream
    assert m[P.FIFO_CTRL3] == 0                                         # only SFLP in the FIFO
    assert e[P.SFLP_ODR] == (0x4B & ~0x38) | (3 << 3)                   # 120 Hz, reserved bits kept
    assert e[P.EMB_FUNC_EN_A] & 2 and e[P.EMB_FUNC_FIFO_EN_A] == 0x02
    assert m[P.FUNC_CFG_ACCESS] == 0                                    # back on the main bank


def test_pi_imu_gravity_readiness_staleness_and_errors():
    dev, clock = FakeLsm6dsv(), Clock()
    imu = P.PiImu(dev=dev, mount=IDENTITY_MOUNT, clock=clock)
    tilt = np.array([math.cos(0.15), 0.0, math.sin(0.15), 0.0])  # 17 deg nose-down about y
    gyro_lsb = (100, -50, 25)
    dev.gyro = struct.pack("<hhh", *gyro_lsb)
    ready = False
    for _ in range(40):
        clock.t += 0.02
        dev.push_quat(tilt)
        dev.push_quat(tilt)
        data, ready = imu.read()
    assert ready
    expect = rotate_inverse(tilt, np.array([0.0, 0.0, -1.0]))
    assert np.allclose(data.gravity, expect, atol=2e-3)
    assert np.allclose(data.gyro, np.array(gyro_lsb) * 0.0175 * math.pi / 180)
    clock.t += 0.15                                   # no SFLP sample for 150 ms
    assert imu.read()[1] is False
    dev.push_quat(tilt)
    assert imu.read()[1] is True
    for _ in range(P.MAX_DRAIN + 5):                  # backlog: reset, do not replay
        dev.push_quat(tilt)
    imu.read()
    assert imu.resets == 1 and not dev.fifo
    dev.fail = True
    assert imu.read()[1] is False


def test_pi_imu_mount_maps_sensor_to_trunk():
    dev, clock = FakeLsm6dsv(), Clock()
    mount = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])  # official zero3
    imu = P.PiImu(dev=dev, mount=mount, clock=clock)
    q_trunk = np.array([1.0, 0.0, 0.0, 0.0])                       # upright
    q_sensor = quat_mul(q_trunk, mount)
    for _ in range(30):
        clock.t += 0.02
        dev.push_quat(q_sensor)
        data, ready = imu.read()
    assert ready and np.allclose(data.gravity, [0, 0, -1], atol=2e-3)


def test_mat2quat_matches_mujoco():
    import mujoco
    from duck_runtime.tools import _mat2quat
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, q)
        got = _mat2quat(M.reshape(3, 3))
        assert np.allclose(got, q if q[0] >= 0 else -q, atol=1e-9)


def test_hip_trim_moves_true_hip_angles_the_way_sim_says():
    import copy
    from duck_runtime.tools import LEFT_HIP_PITCH, RIGHT_HIP_PITCH, hip_trim
    cal = _calibrated()
    old = copy.deepcopy(cal)
    d = math.radians(5.0)  # want the trunk 5 deg further nose-down at HOME
    hip_trim(cal, d)
    for k, expect in ((LEFT_HIP_PITCH, -d), (RIGHT_HIP_PITCH, +d)):
        q = C.DEFAULT_POSITION[k]
        moved = old.joints[k].to_rad(cal.joints[k].to_raw(q)) - q  # true angle change, same command
        assert moved == pytest.approx(expect, abs=2 * math.pi / 4096)
    assert all(a.zero_raw == b.zero_raw for i, (a, b) in enumerate(zip(cal.joints, old.joints))
               if i not in (LEFT_HIP_PITCH, RIGHT_HIP_PITCH))


def test_hip_roll_spread_moves_both_legs_outward_for_the_same_command():
    import copy
    from duck_runtime.tools import LEFT_HIP_ROLL, RIGHT_HIP_ROLL, hip_roll_spread
    cal = _calibrated()
    old = copy.deepcopy(cal)
    s = math.radians(4.0)
    hip_roll_spread(cal, s)
    q = C.DEFAULT_POSITION
    dl = old.joints[LEFT_HIP_ROLL].to_rad(cal.joints[LEFT_HIP_ROLL].to_raw(q[LEFT_HIP_ROLL])) - q[LEFT_HIP_ROLL]
    dr = old.joints[RIGHT_HIP_ROLL].to_rad(cal.joints[RIGHT_HIP_ROLL].to_raw(q[RIGHT_HIP_ROLL])) - q[RIGHT_HIP_ROLL]
    assert dl == pytest.approx(+s, abs=2 * math.pi / 4096)   # left + = outward
    assert dr == pytest.approx(-s, abs=2 * math.pi / 4096)   # right - = outward


def test_imu_snap_recovers_square_mount_and_sign_of_residual():
    from duck_runtime.imu import DEFAULT_MOUNT, SflpDecoder, encode_block
    from duck_runtime.tools import nearest_axis_mount
    true_mount = np.array(DEFAULT_MOUNT)
    tilt = np.array([math.cos(math.radians(1.5)), 0.0, math.sin(math.radians(1.5)), 0.0])  # +3 deg about y
    measured = quat_mul(tilt, true_mount)
    q0, dev = nearest_axis_mount(measured)
    assert abs(abs(np.dot(q0, true_mount)) - 1) < 1e-9
    assert np.degrees(dev[1]) == pytest.approx(3.0, abs=1e-6)
    # with the measured mount an upright trunk reads nose-UP by that residual (pitch = asin(g_x) < 0)
    dec = SflpDecoder(measured)
    for _ in range(30):
        g = dec.decode(encode_block([0, 0, 0], quat_mul(np.array([1.0, 0, 0, 0]), true_mount))).gravity
    assert math.degrees(math.asin(g[0])) == pytest.approx(-3.0, abs=0.2)



# -- the mouth servo (add_mouth.py firmware: 15 servos) ----------------------------------------

STOCK_CONFIG = """#pragma once
#include <stdint.h>
constexpr int FEETECH_RX_PIN = 6;
constexpr int SERVO_COUNT = 14;     // AI-FanGe image: no mouth joint in its 14-action policy
constexpr uint16_t ALL_SERVOS = (1u << SERVO_COUNT) - 1;
constexpr uint8_t DXL_IDS[SERVO_COUNT] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
constexpr uint8_t BUS_IDS[SERVO_COUNT] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
constexpr int8_t JOINT_DIR[SERVO_COUNT] = {-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
constexpr int32_t FT_AT_DXL_CENTER[SERVO_COUNT] = {
  2048,2100,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048
};
constexpr uint16_t FT_MIN[SERVO_COUNT] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
constexpr uint16_t FT_MAX[SERVO_COUNT] = {
  4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095,4095
};
constexpr uint16_t CALIBRATED_MASK = 0x3FFF;
static_assert(SERVO_COUNT <= 14, "Update buffer and diagnostic layouts before adding joints");
"""


def test_add_mouth_patch_keeps_existing_servos(tmp_path):
    import re
    from duck_runtime.esp32_bridge_patch.add_mouth import main as add_mouth
    (tmp_path / "bridge_config.h").write_text(STOCK_CONFIG)
    add_mouth([str(tmp_path)])
    out = (tmp_path / "bridge_config.h").read_text()
    assert "SERVO_COUNT = 15;" in out and "0x7FFF" in out and "SERVO_COUNT <= 16" in out
    assert "{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}" in out
    assert "{-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}" in out          # servo 2's +1 kept
    assert "2048,2100," + ",".join(["2048"] * 13) in out                     # center kept, mouth 2048
    assert (tmp_path / "bridge_config.h.before_mouth").read_text() == STOCK_CONFIG
    add_mouth([str(tmp_path)])                                               # idempotent
    assert (tmp_path / "bridge_config.h").read_text() == out


def _mouth_cal():
    cal = _calibrated()
    m = cal.joints[C.MOUTH_INDEX]
    m.id, m.zero_raw, m.direction, m.calibrated = 15, 1500, 1, True
    return cal


def test_mouth_calibration_needs_the_15_servo_firmware():
    master, slave = pty.openpty()
    fake = FakeBridge(master)                      # stock: 14 servos
    try:
        with pytest.raises(B.BridgeError, match="add_mouth"):
            B.BridgeIO(os.ttyname(slave), _mouth_cal(), StaticImu())
    finally:
        fake.alive = False
        os.close(slave)


def test_mouth_servo_reads_writes_and_takes_part_in_torque_and_fault_clearing():
    from duck_runtime.controller import Controller, Tuning
    from duck_runtime.runner import Runner
    from duck_runtime.safety import Safety

    class _Zero:
        def __call__(self, obs):
            return np.zeros(14, dtype=np.float32)

    master, slave = pty.openpty()
    fake = FakeBridge(master, n_servos=15)
    cal = _mouth_cal()
    io = B.BridgeIO(os.ttyname(slave), cal, StaticImu(), control_dt=0.002)
    try:
        assert io.served == list(range(1, 16))
        m = cal.joints[C.MOUTH_INDEX]
        fake.raw[15] = m.to_raw(C.MOUTH_CLOSED)
        assert io.read().positions[C.MOUTH_INDEX] == pytest.approx(C.MOUTH_CLOSED, abs=2e-3)
        r = Runner(io, Controller(_Zero(), Tuning.official()), Safety(cal.limits()), ramp_s=0.02)
        r.enable()
        assert _eventually(lambda: fake.torque == fake.ALL)  # the mouth holds too
        r.mouth_target = math.radians(20)
        for _ in range(20):
            r.tick()
        assert _eventually(lambda: abs(fake.raw[15] - m.to_raw(math.radians(20))) <= 1)
        r.mouth_target = math.radians(80)                  # clamped to the mouth's range (30 deg)
        r.tick()
        assert _eventually(lambda: abs(fake.raw[15] - m.to_raw(math.radians(30))) <= 1)
        fake.fault = 1
        io.set_torque(False)                               # all 15 off -> clears the fault
        assert _eventually(lambda: fake.fault == 0 and fake.torque == 0)
    finally:
        fake.alive = False
        io.close()



def test_lost_link_limps_and_reconnects_instead_of_crashing():
    from duck_runtime.controller import Controller, Tuning
    from duck_runtime.runner import Runner, State
    from duck_runtime.safety import Safety

    class _Zero:
        def __call__(self, obs):
            return np.zeros(14, dtype=np.float32)

    master, slave = pty.openpty()
    fake = FakeBridge(master)
    cal = _calibrated()
    io = B.BridgeIO(os.ttyname(slave), cal, StaticImu(), control_dt=0.002)
    try:
        r = Runner(io, Controller(_Zero(), Tuning.official()), Safety(cal.limits()), ramp_s=0.02)
        r.enable()
        for _ in range(20):
            r.tick()
        assert r.state is State.HOLD
        fake.mute = True                                  # the ESP32 browns out
        r.tick()                                          # no exception escapes
        assert r.state is State.LIMP and "link lost" in r.refusal and r.link_errors == 1
        fake.mute = False
        fake.torque = 0
        time.sleep(1.05)
        r.tick()                                          # back: reconnected or simply answering again
        assert r.refusal.startswith(("reconnected", "link restored")), r.refusal
        r.tick()
        assert r.state is State.LIMP and r.link_errors >= 1
        r.enable()                                        # usable again
        for _ in range(20):
            r.tick()
        assert r.state is State.HOLD
    finally:
        fake.alive = False
        io.close()


def test_regs_dump_flags_native_d_and_torque_limit():
    from duck_runtime.tools import servo_register_rows
    master, slave = pty.openpty()
    fake = FakeBridge(master, n_servos=3)
    b = B.Bridge(os.ttyname(slave))
    try:
        rows = servo_register_rows(b, b.served_ids())
    finally:
        b.close()
        fake.alive = False
    assert len(rows) == 3
    assert "RAM P/D/I  32/ 40/  0" in rows[0] and "RAM D!=0" in rows[0] and "torque limit<1000" in rows[0]
    master, slave = pty.openpty()
    fake = FakeBridge(master, n_servos=3)
    b = B.Bridge(os.ttyname(slave))
    try:
        idle = servo_register_rows(b, b.served_ids(), torque_mask=0)   # power-up gains with torque off: fine
    finally:
        b.close()
        fake.alive = False
    assert "RAM D!=0" not in idle[0]
