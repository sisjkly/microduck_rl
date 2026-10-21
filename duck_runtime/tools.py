"""Bring-up tools for the real robot. Run on the Pi, robot powered, torque OFF unless stated.

  python -m duck_runtime.tools template  calib.json              # start a calibration file
  python -m duck_runtime.tools scan      --port /dev/ttyUSB0       # who answers on the bus
  python -m duck_runtime.tools live      --port ... --calib calib.json   # angles + IMU, live
  python -m duck_runtime.tools zero      --port ... --calib calib.json   # record zero_raw
  python -m duck_runtime.tools direction --port ... --calib calib.json   # verify signs
  python -m duck_runtime.tools imu-mount --port ... --calib calib.json   # two-pose IMU mount
  python -m duck_runtime.tools home-test --port ... --calib calib.json   # TORQUE: ease to HOME

Order on a new robot: template -> scan -> zero -> direction -> imu-mount -> live (check) ->
home-test (robot hanging) -> run.
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


def _io(args):
    from .feetech import FeetechIO
    return FeetechIO(args.port, Calibration.load(args.calib), echo=args.echo)


def _raw_positions(io) -> dict[int, int]:
    from .feetech import REG_FEEDBACK, FEEDBACK_LEN, _signmag
    raw = io.bus.sync_read(REG_FEEDBACK, FEEDBACK_LEN, io.ids)
    return {i: _signmag(struct.unpack_from("<H", raw[i], 0)[0]) for i in io.ids}


def cmd_template(args):
    Calibration.template().save(args.out)
    print(f"wrote {args.out} (zero_raw=2048, kissqy directions, MJCF limits) — now measure it")


def cmd_scan(args):
    from .feetech import FeetechBus, REG_FEEDBACK, FEEDBACK_LEN, _signmag
    from .contract import JOINT_IDS
    bus = FeetechBus(args.port, echo=args.echo)
    for name, i in [*zip(JOINT_NAMES, JOINT_IDS), ("imu", IMU_ID)]:
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
    bus.close()


def cmd_live(args):
    io = _io(args)
    io.bus.sync_write(40, {i: b"\x00" for i in io.ids})
    try:
        while True:
            s = io.read()
            rows = [f"{n:16s} {math.degrees(q):+7.1f} deg  (HOME {math.degrees(h):+6.1f}, d {math.degrees(q - h):+6.1f})"
                    for n, q, h in zip(JOINT_NAMES, s.positions, DEFAULT_POSITION)]
            g = s.imu.gravity
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
            print("\033[2J\033[H" + "\n".join(rows))
            print(f"gravity(trunk) {np.round(g, 3)}  tilt {tilt:5.1f} deg  gyro {np.round(s.imu.gyro, 2)}  ready={s.imu_ready}")
            print(f"voltage min {io.voltages.min():.1f} V   temp max {io.temperatures.max():.0f} C   (Ctrl+C to quit)")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        io.close()


def cmd_zero(args):
    """Robot in the ZERO pose: legs straight down and parallel, feet flat, neck and head
    straight (every MJCF joint at 0; mouth closed = -5 deg). Use a jig if you have one."""
    cal = Calibration.load(args.calib)
    io = _io(args)
    io.bus.sync_write(40, {i: b"\x00" for i in io.ids})
    print(cmd_zero.__doc__)
    input("Pose the robot, then press Enter to record...")
    samples = [_raw_positions(io) for _ in range(20)]
    only = set(args.joints.split(",")) if args.joints else None
    for j in cal.joints:
        if only and j.name not in only:
            continue
        raw = int(round(np.median([s[j.id] for s in samples])))
        ref = MOUTH_CLOSED if j.name == "mouth" else 0.0
        j.zero_raw = int(round(raw - ref * j.direction * 4096 / (2 * math.pi)))
        print(f"  {j.name:16s} raw {raw:5d} -> zero_raw {j.zero_raw}")
    cal.save(args.calib)
    io.close()
    print(f"saved {args.calib}. Next: direction check.")


def cmd_direction(args):
    cal = Calibration.load(args.calib)
    io = _io(args)
    io.bus.sync_write(40, {i: b"\x00" for i in io.ids})
    for k, j in enumerate(cal.joints):
        if args.joints and j.name not in args.joints.split(","):
            continue
        input(f"\n[{j.name}] 正方向 = {POSITIVE[j.name]}\n  回到零位附近，按回车后在 3 秒内把这个关节往正方向慢慢转 20-30 度…")
        q0 = io.read().positions[k]
        time.sleep(3.0)
        q1 = io.read().positions[k]
        d = math.degrees(q1 - q0)
        if abs(d) < 5:
            print(f"  只动了 {d:+.1f} deg，跳过（重做: --joints {j.name}）")
            continue
        if d > 0:
            print(f"  +{d:.1f} deg  方向正确")
            j.calibrated = True
        else:
            # zero_raw is the count at 0 rad, which does not depend on the sign — except
            # the mouth, recorded at -5 deg: redo `zero --joints mouth` after flipping it.
            j.direction *= -1
            j.calibrated = True
            print(f"  {d:.1f} deg  方向反了 -> direction 改为 {j.direction}"
                  + ("（嘴：请重新 zero --joints mouth）" if j.name == "mouth" else ""))
    cal.save(args.calib)
    io.close()


def _sensor_gravity(io) -> np.ndarray:
    """Gravity in the IMU's own frame (mount ignored), averaged over 1 s."""
    from .imu import SflpDecoder, IDENTITY_MOUNT, rotate_inverse
    dec = SflpDecoder(IDENTITY_MOUNT)
    from .feetech import REG_FEEDBACK, FEEDBACK_LEN
    acc = []
    for _ in range(50):
        raw = io.bus.sync_read(REG_FEEDBACK, FEEDBACK_LEN, [IMU_ID])[IMU_ID]
        d = dec.decode(raw[:12])
        acc.append(rotate_inverse(d.quat, np.array([0.0, 0.0, -1.0])))
        time.sleep(0.02)
    g = np.mean(acc[10:], axis=0)
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
    q = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(q, M.flatten())
    return q if q[0] >= 0 else -q


def cmd_imu_mount(args):
    cal = Calibration.load(args.calib)
    io = _io(args)
    input("姿态 1：机器人直立（躯干水平）静止，按回车…")
    g1 = _sensor_gravity(io)
    input("姿态 2：保持左右水平，把机器人整体向前（低头方向）倾斜 30-60 度并静止，按回车…")
    g2 = _sensor_gravity(io)
    q = mount_from_two_poses(g1, g2)
    cal.imu_mount = [float(v) for v in q]
    cal.imu_mount_verified = True
    cal.save(args.calib)
    print(f"imu_mount = {np.round(q, 5).tolist()}  (official zero3 default is [0.70711, 0, 0.70711, 0])")
    print("用 live 检查：直立时 gravity ≈ [0,0,-1]；前倾时 x 变正；向左倾时 y 变负…实际以 live 显示为准")
    io.close()


def cmd_home_test(args):
    """TORQUE ON. Robot hanging, feet off the ground: ease to HOME over 3 s, hold, report error."""
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = _io(args)
    safety = Safety(cal.limits())
    s = io.read()
    start = s.positions.copy()
    input(f"机器人吊起、脚离地。按回车上力矩并在 3 秒内回到 HOME（Ctrl+C 卸力）…")
    io.set_torque(True)
    try:
        n = 150
        for k in range(n + 100):
            a = min(1.0, k / n)
            t = (1 - a) * start + a * DEFAULT_POSITION
            t[MOUTH_INDEX] = MOUTH_CLOSED
            io.write(safety.clamp(t))
            io.wait_tick()
        s = io.read()
        for n_, q, h in zip(JOINT_NAMES, s.positions, DEFAULT_POSITION):
            flag = "  <-- check" if abs(q - h) > math.radians(5) and n_ != "mouth" else ""
            print(f"  {n_:16s} err {math.degrees(q - h):+6.1f} deg{flag}")
        input("对照实物确认是 STAND 姿态（腿微屈、头抬起）。按回车卸力…")
    finally:
        io.set_torque(False)
        io.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m duck_runtime.tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("template"); p.add_argument("out"); p.set_defaults(fn=cmd_template)
    for name, fn in [("scan", cmd_scan), ("live", cmd_live), ("zero", cmd_zero), ("direction", cmd_direction),
                     ("imu-mount", cmd_imu_mount), ("home-test", cmd_home_test)]:
        p = sub.add_parser(name)
        p.add_argument("--port", required=True)
        p.add_argument("--echo", action="store_true", help="adapter echoes TX bytes")
        if name != "scan":
            p.add_argument("--calib", required=True)
        if name in ("zero", "direction"):
            p.add_argument("--joints", default="", help="comma-separated subset")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
