"""Bring-up tools for the real robot. Run on the Pi, robot powered, torque OFF unless stated.

This robot (default ``--backend bridge``): servos behind the ESP32-S3 MDK2 bridge on
/dev/ttyACM0, LSM6DSV16X on the Pi's I2C bus 1. Stop the old service first:
``sudo systemctl stop microduck-gamepad.service``.

  python -m duck_runtime.tools template  calib.json                # start a calibration file
  python -m duck_runtime.tools scan                                # bridge, 14 servos, IMU
  python -m duck_runtime.tools relax                               # torque off + clear bridge fault
  python -m duck_runtime.tools zero      --calib calib.json        # record zero_raw
  python -m duck_runtime.tools direction --calib calib.json        # verify signs
  python -m duck_runtime.tools imu-mount --calib calib.json        # two-pose IMU mount
  python -m duck_runtime.tools live      --calib calib.json        # angles + IMU, live
  python -m duck_runtime.tools home-test --calib calib.json        # TORQUE: ease to HOME

Order on a new robot: template -> scan -> zero -> direction -> imu-mount -> live (check) ->
home-test (robot hanging) -> run.

``--backend feetech --port /dev/ttyUSB0 [--imu-port ...]`` drives a Feetech bus directly
(FE-URT / FT6 IMU board / ESP32 IMU link) instead.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time

import numpy as np

from .calibration import Calibration
from .contract import DEFAULT_POSITION, IMU_ID, JOINT_NAMES, MOUTH_CLOSED, MOUTH_INDEX

# Positive direction of each joint, as the MJCF defines it (robot facing +x, HOME pose,
# derived by perturbing each joint +0.2 rad in sim). The physical joint must move this
# way when its reading increases.
POSITIVE = {
    "left_hip_yaw": "左腿绕竖直轴转：从上往下看顺时针（左脚尖向内/向右转）",
    "left_hip_roll": "左腿向外侧（向左）张开",
    "left_hip_pitch": "左腿整体向后摆",
    "left_knee": "左小腿/脚向前摆",
    "left_ankle": "左脚尖向下（踮脚方向）",
    "neck_pitch": "脖子向后仰（头整体后移、上抬）",
    "head_pitch": "低头（嘴向前下方）",
    "head_yaw": "头向左转",
    "head_roll": "头向左歪（镜头向左偏）",
    "mouth": "张嘴",
    "right_hip_yaw": "右腿绕竖直轴转：从上往下看顺时针（右脚尖向外/向右转）",
    "right_hip_roll": "右腿向内侧（向左、朝身体中线）收",
    "right_hip_pitch": "右腿整体向前摆",
    "right_knee": "右小腿/脚向后摆",
    "right_ankle": "右脚尖向上（勾脚方向）",
}


# -- backend factory (shared with run.py) ----------------------------------------------------
def add_robot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["bridge", "feetech"], default="bridge",
                   help="bridge: ESP32-S3 MDK2 on USB + Pi I2C IMU (this robot); feetech: direct Feetech bus")
    p.add_argument("--port", default=None, help="default /dev/ttyACM0 (bridge) or /dev/ttyUSB0 (feetech)")
    p.add_argument("--i2c-bus", type=int, default=1, help="bridge backend: Pi I2C bus of the LSM6DSV16X")
    p.add_argument("--echo", choices=["auto", "yes", "no"], default="auto", help="feetech: adapter echoes TX")
    p.add_argument("--imu-port", default=None, help="feetech: ESP32 IMU link port; omit for IMU ID 200 on the bus")


def _port(args) -> str:
    return args.port or ("/dev/ttyACM0" if args.backend == "bridge" else "/dev/ttyUSB0")


def _echo(args):
    return True if args.echo == "yes" else False if args.echo == "no" else "auto"


def make_imu_source(imu_port: str | None, mount):
    """feetech backend: None -> IMU on the servo bus (FT6, ID 200); a port -> ESP32 link."""
    if not imu_port:
        return None
    from .esp32_link import Esp32Imu
    return Esp32Imu(imu_port, mount=mount)


def open_io(args, cal: Calibration):
    if args.backend == "bridge":
        from .bridge import BridgeIO
        from .pi_imu import PiImu
        imu = PiImu(bus=args.i2c_bus, mount=cal.imu_mount)
        try:
            return BridgeIO(_port(args), cal, imu)
        except BaseException:
            imu.close()
            raise
    from .feetech import FeetechIO
    return FeetechIO(_port(args), cal, echo=_echo(args), imu_source=make_imu_source(args.imu_port, cal.imu_mount))


def _io(args):
    return open_io(args, Calibration.load(args.calib))


# -- commands --------------------------------------------------------------------------------
def cmd_template(args):
    if args.layout == "bridge":
        Calibration.template_bridge().save(args.out)
        print(f"wrote {args.out}: bridge IDs (constants.py wiring), direction -1, zero_raw 2048 — now measure it")
    else:
        Calibration.template().save(args.out)
        print(f"wrote {args.out}: official IDs, kissqy directions, zero_raw 2048 — now measure it")


def _scan_bridge(args):
    from .bridge import ADDR_DIAG, ADDR_NPD, BRIDGE_IDS, Bridge, BridgeError, parse_diag
    from .calibration import BRIDGE_IDS as CAL_IDS
    name_of = {i: n for n, i in zip(JOINT_NAMES, CAL_IDS) if i}
    b = Bridge(_port(args))
    try:
        st = b.status()
        print(f"MDK2 v{st['version']}  uptime {st['uptime_ms'] / 1000:.1f} s  reset reason {st['reset_reason']}  "
              f"fault {st['fault']} ({st['fault_name']})")
        print(f"  fresh 0x{st['valid_mask']:04x}  config 0x{st['config_mask']:04x}  torque 0x{st['torque_mask']:04x}  "
              f"fw-calibrated 0x{st['calibrated_mask']:04x}  bus cycle {st['cycle_us']} us  "
              f"incomplete polls {st['incomplete_polls']}  bad frames {st['bad_frames']}")
        npd = b.read(1, ADDR_NPD, 16)
        print(f"  NPD1 (native P/D): {'yes' if npd[:4] == b'NPD1' else 'NO — flash MicroDuck_XGO_r1 firmware'}")
        diag = {i: parse_diag(d) for i, d in b.sync_read(ADDR_DIAG, 32, BRIDGE_IDS).items()}
        for i in BRIDGE_IDS:
            d = diag[i]
            try:
                pd = b.native(i, 50, 2)
                # 41 acc, 44-45 time, 46-47 speed: the bridge writes only 42-43, so whatever sits
                # here limits every move. Training assumed none (kissqy zeroes 44-47).
                mv = b.native(i, 41, 7)
                acc, t_lim, v_lim = mv[0], struct.unpack_from("<H", mv, 3)[0], struct.unpack_from("<H", mv, 5)[0]
                regs = (f"RAM P/D {pd[0]}/{pd[1]}  acc/time/speed {acc}/{t_lim}/{v_lim}"
                        + ("  <-- move limit set" if acc or t_lim or v_lim else ""))
            except BridgeError as e:
                regs = f"native read failed ({e})"
            ok = d["fresh"] and d["configured"] and not d["error"] and d["mode"] == 4
            print(f"  ID {i:2d} {name_of.get(i, '?'):16s} {'ok ' if ok else 'BAD'} raw {d['raw']:5d}  "
                  f"mode {d['mode']}  err 0x{d['error']:02x}  {d['voltage']:4.1f} V  {d['temperature']:2d} C  "
                  f"{regs}  fw map c={d['center']} dir={d['direction']:+d}")
    finally:
        b.close()
    from .pi_imu import PiImu
    from .imu import IDENTITY_MOUNT
    imu = PiImu(bus=args.i2c_bus, mount=IDENTITY_MOUNT)
    try:
        t0, n0 = time.monotonic(), imu.quat_count
        ready = False
        for _ in range(100):
            data, ready = imu.read()
            time.sleep(0.02)
        rate = (imu.quat_count - n0) / (time.monotonic() - t0)
        print(f"IMU LSM6DSV16X @0x{imu.dev.addr:02X} on i2c-{args.i2c_bus}: SFLP {rate:.0f} Hz (expect ~120), "
              f"fifo resets {imu.resets}, ready={ready}, |gyro| {np.linalg.norm(data.gyro):.3f} rad/s (still: ~0)")
    finally:
        imu.close()


def _scan_feetech(args):
    from .feetech import FeetechBus, REG_FEEDBACK, FEEDBACK_LEN, _signmag
    from .contract import JOINT_IDS
    bus = FeetechBus(_port(args), echo=_echo(args))
    devices = list(zip(JOINT_NAMES, JOINT_IDS)) + ([] if args.imu_port else [("imu", IMU_ID)])
    for name, i in devices:
        if not bus.ping(i):
            print(f"  {i:3d} {name:16s} -- no answer")
            continue
        try:
            b = bus.read(i, REG_FEEDBACK, FEEDBACK_LEN)
            p = _signmag(struct.unpack_from("<H", b, 0)[0])
            extra = f"raw pos {p:5d}  {b[6] * 0.1:4.1f} V  {b[7]:2d} C" if name != "imu" else "IMU block OK"
        except Exception as e:  # noqa: BLE001
            extra = f"read failed: {e}"
        print(f"  {i:3d} {name:16s} ok  {extra}")
    print(f"  adapter echo: {bus.echo}   bad packets: {bus.bad_packets}")
    bus.close()
    if args.imu_port:
        from .esp32_link import Esp32Imu
        imu = Esp32Imu(args.imu_port)
        time.sleep(1.0)
        for _ in range(30):
            _, ready = imu.read()
            time.sleep(0.02)
        print(f"  ESP32 IMU on {args.imu_port}: {imu.frames} frames/s, bad crc {imu.parser.bad_crc}, ready={ready}")
        imu.close()


def cmd_scan(args):
    (_scan_bridge if args.backend == "bridge" else _scan_feetech)(args)


def cmd_relax(args):
    if args.backend != "bridge":
        raise SystemExit("relax is for the bridge backend")
    from .bridge import BRIDGE_IDS, Bridge
    b = Bridge(_port(args))
    try:
        b.torque(BRIDGE_IDS, False)
        time.sleep(0.05)
        st = b.status()
        print(f"torque mask 0x{st['torque_mask']:04x}  fault {st['fault']} ({st['fault_name']})")
    finally:
        b.close()


def cmd_live(args):
    io = _io(args)
    io.set_torque(False)
    try:
        while True:
            s = io.read()
            rows = [f"{n:16s} {math.degrees(q):+7.1f} deg  (HOME {math.degrees(h):+6.1f}, d {math.degrees(q - h):+6.1f})"
                    for n, q, h in zip(JOINT_NAMES, s.positions, DEFAULT_POSITION)]
            g = s.imu.gravity
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
            print("\033[2J\033[H" + "\n".join(rows))
            print(f"gravity(trunk) {np.round(g, 3)}  tilt {tilt:5.1f} deg  gyro {np.round(s.imu.gyro, 2)}  ready={s.imu_ready}")
            v = io.voltages[io.voltages > 0]
            print(f"voltage min {v.min() if v.size else 0:.1f} V   temp max {io.temperatures.max():.0f} C   (Ctrl+C to quit)")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        io.close()


def cmd_zero(args):
    """Robot in the ZERO pose: legs straight down and parallel, feet flat, neck and head
    straight (every MJCF joint at 0; mouth closed = -5 deg). Use a jig if you have one."""
    cal = Calibration.load(args.calib)
    io = open_io(args, cal)
    io.set_torque(False)
    print(cmd_zero.__doc__)
    input("Pose the robot, then press Enter to record...")
    samples = [io.raw_positions() for _ in range(20)]
    only = set(args.joints.split(",")) if args.joints else None
    for j in cal.joints:
        if not j.id or (only and j.name not in only):
            continue
        raw = int(round(np.median([s[j.id] for s in samples])))
        ref = MOUTH_CLOSED if j.name == "mouth" else 0.0
        j.zero_raw = int(round(raw - ref * j.direction * 4096 / (2 * math.pi)))
        print(f"  {j.name:16s} ID {j.id:2d} raw {raw:5d} -> zero_raw {j.zero_raw}")
    cal.save(args.calib)
    io.close()
    print(f"saved {args.calib}. Next: direction check.")


def cmd_direction(args):
    cal = Calibration.load(args.calib)
    io = open_io(args, cal)
    io.set_torque(False)
    for k, j in enumerate(cal.joints):
        if not j.id or (args.joints and j.name not in args.joints.split(",")):
            continue
        input(f"\n[{j.name} / ID {j.id}] 正方向 = {POSITIVE[j.name]}\n  回到零位附近，按回车后在 3 秒内把这个关节往正方向慢慢转 20-30 度…")
        q0 = io.read().positions[k]
        time.sleep(3.0)
        q1 = io.read().positions[k]
        d = math.degrees(q1 - q0)
        if abs(d) < 5:
            print(f"  只动了 {d:+.1f} deg，跳过（重做: --joints {j.name}）")
            continue
        if d > 0:
            print(f"  +{d:.1f} deg  方向正确")
        else:
            # zero_raw is the count at 0 rad, which does not depend on the sign — except
            # the mouth, recorded at -5 deg: redo `zero --joints mouth` after flipping it.
            j.direction *= -1
            print(f"  {d:.1f} deg  方向反了 -> direction 改为 {j.direction}"
                  + ("（嘴：请重新 zero --joints mouth）" if j.name == "mouth" else ""))
        j.calibrated = True
    cal.save(args.calib)
    io.close()


def _sensor_gravity(io) -> np.ndarray:
    """Gravity in the IMU's own frame (mount ignored), averaged over 1 s."""
    from .imu import SflpDecoder, IDENTITY_MOUNT, rotate_inverse
    dec = SflpDecoder(IDENTITY_MOUNT)
    acc = []
    for _ in range(50):
        block = io.raw_imu_block()
        if block is not None:
            d = dec.decode(block)
            if dec.quat_samples:
                acc.append(rotate_inverse(d.quat, np.array([0.0, 0.0, -1.0])))
        time.sleep(0.02)
    if len(acc) < 20:
        raise RuntimeError("IMU gives no quaternion: check wiring (tools scan)")
    g = np.mean(acc[len(acc) // 4:], axis=0)
    return g / np.linalg.norm(g)


def mount_from_two_poses(g_up: np.ndarray, g_nose_down: np.ndarray) -> np.ndarray:
    """Sensor->trunk rotation from gravity seen in the sensor frame in two poses:
    upright (trunk gravity = [0,0,-1]) and pitched nose-down (trunk gravity has +x)."""
    z_t = -g_up / np.linalg.norm(g_up)                  # trunk +z in sensor coordinates
    x_t = g_nose_down - np.dot(g_nose_down, z_t) * z_t  # forward = where gravity tips to
    if np.linalg.norm(x_t) < 0.2:
        raise ValueError("second pose not tilted enough (need >= ~15 deg nose-down)")
    x_t /= np.linalg.norm(x_t)
    y_t = np.cross(z_t, x_t)
    M = np.stack([x_t, y_t, z_t])                       # v_trunk = M @ v_sensor
    return _mat2quat(M)


def _mat2quat(M: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z], w >= 0 (Shepperd; no mujoco on the Pi)."""
    t = np.trace(M)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        q = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(M)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + M[i, i] - M[j, j] - M[k, k])
        q = [0.0] * 4
        q[0] = (M[k, j] - M[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (M[j, i] + M[i, j]) / s
        q[1 + k] = (M[k, i] + M[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def cmd_imu_mount(args):
    cal = Calibration.load(args.calib)
    io = open_io(args, cal)
    io.set_torque(False)
    input("姿态 1：机器人直立（躯干水平）静止，按回车…")
    g1 = _sensor_gravity(io)
    input("姿态 2：保持左右水平，把机器人整体向前（低头方向）倾斜 30-60 度并静止，按回车…")
    g2 = _sensor_gravity(io)
    q = mount_from_two_poses(g1, g2)
    cal.imu_mount = [float(v) for v in q]
    cal.imu_mount_verified = True
    cal.save(args.calib)
    print(f"imu_mount = {np.round(q, 5).tolist()}  (official zero3 default is [0.70711, 0, 0.70711, 0])")
    print("用 live 检查：直立时 gravity ≈ [0,0,-1]；前倾时 x 变正；向左倾时 y 变正")
    io.close()


def cmd_home_test(args):
    """TORQUE ON. Robot hanging, feet off the ground: ease to HOME over 3 s, hold, report error."""
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    safety = Safety(cal.limits())
    io.set_torque(False)
    s = io.read()
    start = s.positions.copy()
    input("机器人吊起、脚离地。按回车上力矩并在 3 秒内回到 HOME（Ctrl+C 卸力）…")
    io.set_torque(True)
    try:
        n = 150
        for k in range(n + 100):
            a = min(1.0, k / n)
            t = (1 - a) * start + a * DEFAULT_POSITION
            t[MOUTH_INDEX] = MOUTH_CLOSED
            io.read()  # keeps the bridge's host watchdog fed and checks feedback freshness
            io.write(safety.clamp(t))
            io.wait_tick()
        s = io.read()
        for j, q, h in zip(cal.joints, s.positions, DEFAULT_POSITION):
            if not j.id:
                continue
            flag = "  <-- check" if abs(q - h) > math.radians(5) else ""
            print(f"  {j.name:16s} err {math.degrees(q - h):+6.1f} deg{flag}")
        input("对照实物确认是 STAND 姿态（腿微屈、头抬起）。按回车卸力…")
    finally:
        io.set_torque(False)
        io.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m duck_runtime.tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("template")
    p.add_argument("out")
    p.add_argument("--layout", choices=["bridge", "official"], default="bridge")
    p.set_defaults(fn=cmd_template)
    for name, fn in [("scan", cmd_scan), ("relax", cmd_relax), ("live", cmd_live), ("zero", cmd_zero),
                     ("direction", cmd_direction), ("imu-mount", cmd_imu_mount), ("home-test", cmd_home_test)]:
        p = sub.add_parser(name)
        add_robot_args(p)
        if name not in ("scan", "relax"):
            p.add_argument("--calib", required=True)
        if name in ("zero", "direction"):
            p.add_argument("--joints", default="", help="comma-separated subset")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
