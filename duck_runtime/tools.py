"""Bring-up tools for the real robot. Run on the Pi, robot powered, torque OFF unless stated.

This robot (default ``--backend bridge``): servos behind the ESP32-S3 MDK2 bridge on
/dev/ttyACM0, LSM6DSV16X on the Pi's I2C bus 1. Stop the old service first:
``sudo systemctl stop microduck-gamepad.service``.

  python -m duck_runtime.tools template  calib.json                # start a calibration file
  python -m duck_runtime.tools scan                                # bridge, 14 servos, IMU
  python -m duck_runtime.tools relax                               # torque off + clear bridge fault
  python -m duck_runtime.tools step-test --id 13                   # TORQUE, one servo: step response
  python -m duck_runtime.tools zero      --calib calib.json        # record zero_raw
  python -m duck_runtime.tools direction --calib calib.json        # verify signs
  python -m duck_runtime.tools imu-mount --calib calib.json        # two-pose IMU mount
  python -m duck_runtime.tools live      --calib calib.json        # angles + IMU, live
  python -m duck_runtime.tools home-test --calib calib.json        # TORQUE: ease to HOME
  python -m duck_runtime.tools trim      --calib calib.json        # TORQUE, standing: hip-pitch trim
  python -m duck_runtime.tools hip-roll  --calib calib.json        # TORQUE, standing: hip-roll preload sweep
  python -m duck_runtime.tools imu-snap  --calib calib.json        # square IMU board: snap mount to axis-aligned
  python -m duck_runtime.tools pose-check --calib calib.json       # TORQUE, hanging: robot moves, you judge signs
  python -m duck_runtime.tools sign-test --calib calib.json        # TORQUE, standing: IMU judges knee/hip-yaw signs

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
    "neck_pitch": "脖子向后仰（整个头往后移，嘴那一侧略抬起）",
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
    name_of.setdefault(15, "mouth")
    b = Bridge(_port(args))
    try:
        st = b.status()
        rr = {0: "unknown", 1: "power-on", 3: "software", 4: "crash", 5: "int watchdog", 6: "task watchdog",
              7: "watchdog", 9: "BROWNOUT (supply dip)", 11: "USB", 12: "JTAG/USB flash"}.get(st["reset_reason"], "?")
        print(f"MDK2 v{st['version']}  uptime {st['uptime_ms'] / 1000:.1f} s  reset reason {st['reset_reason']} ({rr})  "
              f"fault {st['fault']} ({st['fault_name']})")
        print(f"  fresh 0x{st['valid_mask']:04x}  config 0x{st['config_mask']:04x}  torque 0x{st['torque_mask']:04x}  "
              f"fw-calibrated 0x{st['calibrated_mask']:04x}  bus cycle {st['cycle_us']} us  "
              f"incomplete polls {st['incomplete_polls']}  bad frames {st['bad_frames']}")
        npd = b.read(1, ADDR_NPD, 16)
        print(f"  NPD1 (native P/D): {'yes' if npd[:4] == b'NPD1' else 'NO — flash MicroDuck_XGO_r1 firmware'}")
        served = b.served_ids()
        print(f"  bridge serves IDs {served}" + ("  (mouth firmware)" if 15 in served else
              "  (no mouth: duck_runtime/esp32_bridge_patch/add_mouth.py adds ID 15)"))
        diag = {i: parse_diag(d) for i, d in b.sync_read(ADDR_DIAG, 32, served).items()}
        for i in served:
            d = diag[i]
            try:
                pd = b.native(i, 50, 2)
                # 41 acc, 44-45 time, 46-47 speed: the bridge writes only 42-43, so whatever sits
                # here limits every move. Training assumed none (kissqy zeroes 44-47).
                mv = b.native(i, 41, 7)
                acc, t_lim, v_lim = mv[0], struct.unpack_from("<H", mv, 3)[0], struct.unpack_from("<H", mv, 5)[0]
                lo, hi = struct.unpack("<HH", b.native(i, 9, 4))  # EEPROM angle limits: the servo clamps goals
                regs = (f"RAM P/D {pd[0]}/{pd[1]}  acc/time/speed {acc}/{t_lim}/{v_lim}  angle limit {lo}..{hi}"
                        + ("  <-- angle limit" if (lo or hi) and (lo, hi) != (0, 4095) else ""))
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
        b.torque(b.served_ids(), False)
        time.sleep(0.05)
        st = b.status()
        print(f"torque mask 0x{st['torque_mask']:04x}  fault {st['fault']} ({st['fault_name']})")
    finally:
        b.close()


def servo_register_rows(b, ids, torque_mask: int | None = None) -> list[str]:
    """One native read of HD1910 registers 21..52 per servo (read-only, works with torque on).
    21-23 EEPROM P/D/I, 26/27 dead zone, 41 acceleration, 46-47 goal speed, 48-49 torque
    limit, 50-52 RAM P/D/I (what the servo runs with). kissqy control.43 fixed his body jitter
    by writing RAM D=0 and torque limit 1000 (duck-control/src/hd1910.rs write_fixed_settings)."""
    rows = []
    for i in ids:
        try:
            r = b.native(i, 21, 32)
        except Exception as e:          # noqa: BLE001 - one slow servo should not hide the rest
            rows.append(f"  ID {i:2d}: read failed ({e})")
            continue
        at = lambda a: r[a - 21]
        u16 = lambda a: struct.unpack_from("<H", r, a - 21)[0]
        flag = []
        on = torque_mask is None or bool(torque_mask >> (i - 1) & 1)
        if at(51) != 0 and on:              # torque off: the power-up gains, restored on exit
            flag.append("RAM D!=0")
        if u16(48) < 1000:
            flag.append("torque limit<1000")
        rows.append(f"  ID {i:2d}: RAM P/D/I {at(50):3d}/{at(51):3d}/{at(52):3d}  EEPROM P/D/I {at(21):3d}/{at(22):3d}/{at(23):3d}"
                    f"  dead zone {at(26)}/{at(27)}  acc {at(41):3d}  goal speed {u16(46):4d}  torque limit {u16(48):4d}"
                    + (f"   <- {', '.join(flag)}" if flag else ""))
    return rows


def cmd_regs(args):
    """Read-only register dump of every served servo (safe while the robot stands)."""
    if args.backend != "bridge":
        raise SystemExit("regs is for the bridge backend")
    from .bridge import Bridge
    b = Bridge(_port(args))
    try:
        ids = b.served_ids()
        mask = b.status()["torque_mask"]
        print(f"served IDs {list(ids)}, torque mask 0x{mask:04x}")
        for line in servo_register_rows(b, ids, mask):
            print(line)
        print("Torque off -> RAM shows the power-up gains (HD1910: 32/40): duck_runtime writes P=5 D=0 and reads\n"
              "them back before every torque-on (bridge.py set_torque: refuses torque otherwise), and restores the\n"
              "power-up gains when it exits. The torque limit is never written: it stays as the servo powered up.")
    finally:
        b.close()


# step-test reference numbers come from SimIO (BAM HD1910, P=5 D=0, no extra delay), robot
# standing at HOME, +150 counts on one joint.
def cmd_step_test(args):
    """TORQUE ON ONE SERVO (default ID 13 head_yaw), every other joint limp: support the robot
    or lay it on its back. P/D 5/0 like training, hold 0.5 s, step +N counts, log 1 s."""
    if args.backend != "bridge":
        raise SystemExit("step-test is for the bridge backend")
    from .bridge import ADDR_DIAG, BRIDGE_IDS, Bridge, parse_diag
    i, n = args.id, args.counts
    b = Bridge(_port(args))
    regs = b.native(i, 41, 9)
    pd0 = b.native(i, 50, 2)
    print(f"ID {i}: acc {regs[0]}  44-45 {struct.unpack_from('<H', regs, 3)[0]}  46-47 {struct.unpack_from('<H', regs, 5)[0]}"
          f"  torque limit(48) {struct.unpack_from('<H', regs, 7)[0]}  RAM P/D {pd0[0]}/{pd0[1]}")
    input(f"只给 ID {i} 上力矩（P=5 D=0），其他关节无力：先扶好/放倒机器人。回车开始…")
    st = b.status()
    if st["fault"]:
        b.close()
        raise SystemExit(f"bridge fault {st['fault']} ({st['fault_name']}): run tools relax first")

    def diag():
        return parse_diag(b.read(i, ADDR_DIAG, 32))

    try:
        b.npd(i, 5, 0)
        t_end = time.monotonic() + 0.5
        while b.native(i, 50, 2) != bytes([5, 0]):
            if time.monotonic() > t_end:
                raise SystemExit("P/D readback failed")
        d = diag()
        dx = lambda ft: 2048 + d["direction"] * (ft - d["center"])  # noqa: E731
        start = d["raw"]
        b.sync_write(116, 4, {i: struct.pack("<i", dx(start))})
        b.torque([i], True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            diag()
            time.sleep(0.005)
        hold = diag()["raw"]
        b.sync_write(116, 4, {i: struct.pack("<i", dx(start + n))})
        t0 = time.monotonic()
        log = []
        while time.monotonic() - t0 < 1.0:
            log.append((time.monotonic() - t0, diag()["raw"]))
            time.sleep(0.002)  # the bridge refreshes feedback every ~6-10 ms anyway
    finally:
        b.torque(b.served_ids(), False)
        time.sleep(0.05)
        try:
            b.npd(i, pd0[0], pd0[1])
        finally:
            b.close()
    t = np.array([x[0] for x in log])
    frac = (np.array([x[1] for x in log]) - hold) / n
    def first(level):
        k = np.nonzero(frac >= level)[0]
        return f"{t[k[0]] * 1000:.0f} ms" if k.size else "never"
    print(f"hold drift {hold - start:+d} cnt; step +{n} cnt: t50 {first(0.5)}  t90 {first(0.9)}  "
          f"final {frac[-1]:.2f}  peak {frac.max():.2f}  ({len(log)} samples)")
    for ms in (20, 40, 60, 80, 100, 150, 200, 300, 500, 800):
        k = min(np.searchsorted(t, ms / 1000), len(t) - 1)
        print(f"  {ms:4d} ms  {frac[k]:5.2f}")
    print("sim (BAM HD1910 P=5, 150 cnt): head_yaw t50 80 ms, head_pitch t50 75 / t90 150 ms, left_knee t50 100 / t90 380 ms")


def cmd_live(args):
    io = _io(args)
    io.set_torque(False)
    try:
        while True:
            s = io.read()
            diag = getattr(io, "diag", {})  # bridge: raw HD1910 counts, to debug calibration
            rows = [f"{j.name:16s} {math.degrees(q):+7.1f} deg  (HOME {math.degrees(h):+6.1f}, d {math.degrees(q - h):+6.1f})"
                    + (f"  raw {diag[j.id]['raw']:5d} zero {j.zero_raw:4d} dir {j.direction:+d}"
                       f"{' ERR 0x%02x' % diag[j.id]['error'] if diag[j.id]['error'] else ''}" if j.id in diag else "")
                    for j, q, h in zip(io.cal.joints, s.positions, DEFAULT_POSITION)]
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
    if args.joints == "mouth":
        input("用手把嘴完全闭上（闭合位置记为 -5 度），按回车记录…")
    else:
        print(cmd_zero.__doc__)
        input("Pose the robot, then press Enter to record...")
    samples, grav = [], []
    for _ in range(45):  # also warms the IMU decoder; trunk attitude tells whether the pose was level
        s_ = io.read()
        samples.append(io.raw_positions())
        if s_.imu_ready:
            grav.append(s_.imu.gravity)
        time.sleep(0.02)
    if grav:
        g = np.mean(grav, axis=0)
        pitch, roll = math.degrees(math.asin(max(-1, min(1, g[0])))), math.degrees(math.asin(max(-1, min(1, g[1]))))
        print(f"  trunk at record: pitch {pitch:+.1f} deg (+ = nose down), roll {roll:+.1f} deg"
              + ("   <-- trunk not level: hip zeros absorb this tilt" if abs(pitch) > 1.5 or abs(roll) > 1.5 else ""))
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


def cmd_mouth(args):
    """Add the mouth servo to calib.json (after flashing the 15-servo bridge firmware,
    esp32_bridge_patch/add_mouth.py): set its ID, find its direction (open = +), record
    its zero with the mouth closed. Torque stays off throughout."""
    cal = Calibration.load(args.calib)
    j = cal.joints[MOUTH_INDEX]
    if j.id and j.id != args.id:
        print(f"mouth was ID {j.id}; now {args.id}")
    j.id, j.calibrated = args.id, False
    cal.save(args.calib)
    print(f"mouth = ID {args.id} in {args.calib}")
    args.joints = "mouth"
    cmd_direction(args)
    if not Calibration.load(args.calib).joints[MOUTH_INDEX].calibrated:
        raise SystemExit("嘴的方向没测出来：重跑 tools mouth")
    cmd_zero(args)
    print("嘴标定完成。网页中控的「张嘴」滑条 / 终端按 m 可以开合。")


def cmd_direction(args):
    cal = Calibration.load(args.calib)
    io = open_io(args, cal)
    io.set_torque(False)
    for k, j in enumerate(cal.joints):
        if not j.id or (args.joints and j.name not in args.joints.split(",")):
            continue
        input(f"\n[{j.name} / ID {j.id}] 正方向 = {POSITIVE[j.name]}\n  回到零位附近，按回车后在 3 秒内把这个关节往正方向转 10 度以上（髋横滚活动范围小，10 度即可）…")
        q0 = io.read().positions[k]
        # Largest excursion over 3 s, so springing back or releasing early does not hide it.
        t_end, d = time.monotonic() + 3.0, 0.0
        while time.monotonic() < t_end:
            e = math.degrees(io.read().positions[k] - q0)
            d = e if abs(e) > abs(d) else d
            time.sleep(0.03)
        if abs(d) < 4:
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


def nearest_axis_mount(q) -> tuple[np.ndarray, np.ndarray]:
    """Closest of the 24 axis-aligned sensor->trunk rotations, and the measured mount's
    deviation from it as a trunk-frame rotation vector (rad: about x, y, z)."""
    import itertools
    from .imu import quat_mul
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (k, sgn) in enumerate(zip(perm, signs)):
                M[i, k] = sgn
            if np.linalg.det(M) < 0:
                continue
            c = _mat2quat(M)
            if best is None or abs(np.dot(c, q)) > abs(np.dot(best, q)):
                best = c
    if np.dot(best, q) < 0:
        q = -q
    dev = quat_mul(q, best * np.array([1, -1, -1, -1]))
    dev = dev if dev[0] >= 0 else -dev
    n = float(np.linalg.norm(dev[1:]))
    return best, (2 * math.atan2(n, dev[0]) / n * dev[1:]) if n > 1e-12 else np.zeros(3)


def cmd_imu_snap(args):
    """If the IMU board is screwed square to the trunk (like the official board), its true
    mount is axis-aligned and the two-pose measurement's small residual is posing error.
    Snap imu_mount to the nearest axis-aligned rotation."""
    cal = Calibration.load(args.calib)
    q0, dev = nearest_axis_mount(cal.imu_mount)
    d = np.degrees(dev)
    print(f"measured imu_mount {np.round(cal.imu_mount, 4).tolist()}")
    print(f"nearest axis-aligned {np.round(q0, 4).tolist()}; residual roll {d[0]:+.2f}  pitch {d[1]:+.2f}  yaw {d[2]:+.2f} deg")
    from .imu import SflpDecoder, encode_block
    dec = SflpDecoder(cal.imu_mount)  # what an upright trunk reads with the measured mount
    for _ in range(SflpDecoder.READY_SAMPLES + 3):
        g = dec.decode(encode_block([0, 0, 0], q0)).gravity
    print(f"(a square board on an upright trunk would currently read pitch {math.degrees(math.asin(g[0])):+.1f} deg "
          f"(+ = nose down), roll {math.degrees(math.asin(g[1])):+.1f} deg instead of 0/0)")
    if np.linalg.norm(d) > 10:
        raise SystemExit("residual > 10 deg: the board is not square to the trunk (or the measurement is off); not snapping")
    if input("IMU 板是方正地固定在躯干上的吗？输入 y 回车改用方正安装姿态：").strip().lower() == "y":
        cal.imu_mount = [float(v) for v in q0]
        cal.imu_mount_verified = True
        cal.save(args.calib)
        print(f"saved {args.calib}. 接着跑 trim（髋俯仰）让 HOME 站姿重新对上。")


def cmd_pose_check(args):
    """TORQUE ON, robot HANGING (feet off the ground). From HOME, moves one joint at a time
    +15 deg and asks whether it moved the way the MJCF says positive is. Unlike `direction`
    (you move the joint, the encoder judges), here the robot moves and you judge -- it also
    catches joints whose HOME value is ~0 (knees, hip yaw, head yaw/roll), which home-test
    cannot reveal. Answer n to flip that joint's direction (zero_raw is unaffected)."""
    import select
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    safety = Safety(cal.limits())
    home = DEFAULT_POSITION.copy()
    home[MOUTH_INDEX] = MOUTH_CLOSED
    io.set_torque(False)
    start = io.read().positions.copy()
    input("机器人吊起、脚离地，按回车上力矩回 HOME…")
    io.set_torque(True)
    flips = []

    def glide(a, b, n):
        for k in range(n):
            x = (k + 1) / n
            io.read()
            io.write(safety.clamp((1 - x) * a + x * b))
            io.wait_tick()

    def hold_until_line(t):
        while not select.select([sys.stdin], [], [], 0)[0]:
            io.read()
            io.write(safety.clamp(t))
            io.wait_tick()
        return sys.stdin.readline().strip().lower()

    try:
        glide(start, home, 100)
        only = set(args.joints.split(",")) if args.joints else None
        for k, j in enumerate(cal.joints):
            if not j.id or (only and j.name not in only):
                continue
            t = home.copy()
            t[k] += math.radians(15)
            glide(home, t, 25)
            print(f"\n[{j.name} / ID {j.id}] 刚才应该是：{POSITIVE[j.name]}（约 15 度）。对吗？y 回车=对，n 回车=反了，r 回车=再动一次：", flush=True)
            while True:
                ans = hold_until_line(t)
                if ans != "r":
                    break
                glide(t, home, 25)
                glide(home, t, 25)
                print("  再看一次，y / n / r：", flush=True)
            if ans == "n":
                flips.append(j.name)
            glide(t, home, 25)
    finally:
        io.set_torque(False)
        io.close()
    for j in cal.joints:
        if j.name in flips:
            j.direction *= -1
    if flips:
        cal.save(args.calib)
        print(f"flipped {flips}; saved {args.calib}. 方向改了：重新跑 home-test 确认 HOME 姿态。")
    else:
        print("所有检查的关节方向都对。")


def cmd_sign_test(args):
    """TORQUE ON, robot STANDING on flat ground at HOME (hand right next to it). Objective
    sign check for the joints that sit at ~0 in HOME, judged by the IMU, not by eye:
      twist: both hip yaws +12 deg (model sense). SimIO: trunk yaws +3.9 deg if both signs
             are right, -4.2 if both are reversed, ~0 if one is.
      knees: both knees +4 deg (model sense: shins forward). SimIO: trunk pitches forward
             ~+8 deg if right; ~0 if reversed.
    Prints a verdict; changes nothing."""
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    safety = Safety(cal.limits())
    home = DEFAULT_POSITION.copy()
    home[MOUTH_INDEX] = MOUTH_CLOSED
    io.set_torque(False)
    start = io.read().positions.copy()
    input("鸭子放在平地上（先扶着），按回车上力矩回 HOME；站稳后手放在旁边准备扶，但不要碰它…")
    io.set_torque(True)
    out = {}

    def glide(a, b, n, yaw=None):
        g = None
        for k in range(n):
            x = (k + 1) / n
            s_ = io.read()
            io.write(safety.clamp((1 - x) * a + x * b))
            if yaw is not None and s_.imu_ready:
                yaw[0] += math.degrees(s_.imu.gyro[2]) * io.control_dt
            g = s_.imu.gravity
            io.wait_tick()
        return g

    def pitch(g):
        return math.degrees(math.asin(max(-1.0, min(1.0, g[0]))))

    try:
        glide(start, home, 100)
        glide(home, home, 100)
        # twist
        t = home.copy()
        t[0] += math.radians(12)
        t[10] += math.radians(12)
        yaw = [0.0]
        glide(home, t, 50, yaw)
        glide(t, t, 50, yaw)
        out["twist"] = yaw[0]
        glide(t, home, 50)
        glide(home, home, 75)
        # knees
        g0 = glide(home, home, 25)
        t = home.copy()
        t[3] += math.radians(4)
        t[13] -= math.radians(4)
        glide(home, t, 50)
        g1 = glide(t, t, 50)
        out["knees"] = pitch(g1) - pitch(g0)
        glide(t, home, 50)
        glide(home, home, 50)
    finally:
        io.set_torque(False)
        io.close()
    tw, kn = out.get("twist"), out.get("knees")
    print(f"twist: trunk yaw {tw:+.1f} deg  (sim: +3.9 both right / -4.2 both reversed / ~0 one reversed)")
    print("   -> " + ("hip yaw signs RIGHT" if tw > 2 else "BOTH hip yaws REVERSED" if tw < -2 else "one hip yaw reversed or inconclusive"))
    print(f"knees: trunk pitch {kn:+.1f} deg  (sim: about +8 if right, ~0 if reversed)")
    print("   -> " + ("knee signs RIGHT" if kn > 3 else "knees REVERSED" if abs(kn) < 1.5 else "inconclusive"))


def cmd_home_test(args):
    """TORQUE ON. Robot hanging, feet off the ground: ease to HOME over 3 s, hold, report error."""
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    safety = Safety(cal.limits())
    home = DEFAULT_POSITION.copy()
    home[5:9] += [math.radians(float(v)) for v in args.head_cmd.split(",")]  # neck/head offsets
    io.set_torque(False)
    s = io.read()
    start = s.positions.copy()
    input("机器人吊起、脚离地。按回车上力矩并在 3 秒内回到 HOME（Ctrl+C 卸力）…")
    io.set_torque(True)
    try:
        n = 150
        for k in range(n + 100):
            a = min(1.0, k / n)
            t = (1 - a) * start + a * home
            t[MOUTH_INDEX] = MOUTH_CLOSED
            io.read()  # keeps the bridge's host watchdog fed and checks feedback freshness
            io.write(safety.clamp(t))
            io.wait_tick()
        s = io.read()
        for j, q, h in zip(cal.joints, s.positions, home):
            if not j.id:
                continue
            flag = "  <-- check" if abs(q - h) > math.radians(5) else ""
            print(f"  {j.name:16s} err {math.degrees(q - h):+6.1f} deg{flag}")
        # Keep commanding while waiting: the bridge drops torque after 1 s without host traffic.
        print("对照实物确认是 STAND 姿态（腿微屈、头抬起）。按回车卸力…", flush=True)
        import select
        while not select.select([sys.stdin], [], [], 0)[0]:
            io.read()
            io.write(safety.clamp(t))
            io.wait_tick()
        sys.stdin.readline()
    finally:
        io.set_torque(False)
        io.close()


# Trunk pitch of the simulated robot holding HOME on flat ground (SimIO: 813 g + BAM HD1910,
# P=5): what the policy saw in training when it stood at HOME. Measured 2026-10-21.
HOME_STAND_PITCH_DEG = 2.5
LEFT_HIP_PITCH, RIGHT_HIP_PITCH = 2, 12


def hip_trim(cal: Calibration, delta_pitch_rad: float) -> None:
    """Shift both hip-pitch zeros so the same HOME command tilts the trunk delta further
    nose-down (feet flat). Sign from SimIO: true left hip +e / right hip -e pitches the
    trunk nose-UP, so nose-down by delta needs true left -delta, right +delta; a joint whose
    true angle must move by e for the same command gets zero_raw += e * counts * direction."""
    c = 4096.0 / (2 * math.pi)
    for k, e in ((LEFT_HIP_PITCH, -delta_pitch_rad), (RIGHT_HIP_PITCH, +delta_pitch_rad)):
        j = cal.joints[k]
        j.zero_raw = int(round(j.zero_raw + e * c * j.direction))


def cmd_trim(args):
    """TORQUE ON, robot STANDING on flat ground at HOME (hand near it, not pushing).
    Measures trunk pitch over 3 s and offers to trim the hip-pitch zeros so that HOME stands
    at the pitch the policy was trained with (HOME_STAND_PITCH_DEG)."""
    import select
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    safety = Safety(cal.limits())
    home = DEFAULT_POSITION.copy()
    home[MOUTH_INDEX] = MOUTH_CLOSED
    io.set_torque(False)
    start = io.read().positions.copy()
    input("鸭子放在平地上（先扶着），按回车上力矩并回到 HOME；站稳后手留在旁边但不要碰它…")
    io.set_torque(True)
    try:
        grav, n_ramp, n_settle, n_meas = [], 100, 100, 150
        for k in range(n_ramp + n_settle + n_meas):
            a = min(1.0, k / n_ramp)
            s = io.read()
            io.write(safety.clamp((1 - a) * start + a * home))
            if k >= n_ramp + n_settle and s.imu_ready and np.linalg.norm(s.imu.gyro) < 0.3:
                grav.append(s.imu.gravity)
            io.wait_tick()
        if len(grav) < 50:
            print("机器人在晃或 IMU 未就绪，没测到稳定数据；扶稳后重试")
            return
        g = np.mean(grav, axis=0)
        pitch, roll = math.degrees(math.asin(g[0])), math.degrees(math.asin(g[1]))
        target = HOME_STAND_PITCH_DEG if args.target is None else args.target
        delta = target - pitch
        print(f"HOME 站立：俯仰 {pitch:+.1f} deg（+ 低头），横滚 {roll:+.1f} deg；目标 {target:+.1f} deg"
              f"（训练时 HOME 站立约 {HOME_STAND_PITCH_DEG:+.1f}）")
        if abs(delta) < 0.7:
            print("差值 < 0.7 deg，不用修正。按回车卸力。")
            apply = False
        else:
            print(f"建议把左右髋俯仰零位修正 {delta:+.1f} deg（躯干往{'前' if delta > 0 else '后'}）。输入 y 回车应用，直接回车放弃：", flush=True)
        while not select.select([sys.stdin], [], [], 0)[0]:
            io.read()
            io.write(safety.clamp(home))
            io.wait_tick()
        apply = abs(delta) >= 0.7 and sys.stdin.readline().strip().lower() == "y"
    finally:
        io.set_torque(False)
        io.close()
    if apply:
        before = (cal.joints[LEFT_HIP_PITCH].zero_raw, cal.joints[RIGHT_HIP_PITCH].zero_raw)
        hip_trim(cal, math.radians(delta))
        cal.save(args.calib)
        print(f"hip pitch zero_raw L {before[0]} -> {cal.joints[LEFT_HIP_PITCH].zero_raw}, "
              f"R {before[1]} -> {cal.joints[RIGHT_HIP_PITCH].zero_raw}; saved {args.calib}. 再跑一次 trim 确认。")


LEFT_HIP_ROLL, RIGHT_HIP_ROLL = 1, 11


def hip_roll_spread(cal: Calibration, spread_rad: float) -> None:
    """Shift both hip-roll zeros so the same command puts each leg spread_rad further OUT
    (left hip roll + = outward, right hip roll + = inward)."""
    c = 4096.0 / (2 * math.pi)
    for k, e in ((LEFT_HIP_ROLL, +spread_rad), (RIGHT_HIP_ROLL, -spread_rad)):
        j = cal.joints[k]
        j.zero_raw = int(round(j.zero_raw + e * c * j.direction))


def cmd_hip_roll(args):
    """TORQUE ON, robot STANDING at HOME (hand near, not pushing). Spreads both legs in/out
    by a few degrees around HOME and reads hip-roll current and trunk roll at each step.
    Correct hip-roll zeros stand with the legs relaxed: low current on both, roll ~ 0."""
    import select
    from .safety import Safety
    cal = Calibration.load(args.calib)
    cal.validate()
    io = open_io(args, cal)
    if not hasattr(io, "currents"):
        raise SystemExit("needs the bridge backend (current feedback)")
    safety = Safety(cal.limits())
    home = DEFAULT_POSITION.copy()
    home[MOUTH_INDEX] = MOUTH_CLOSED
    io.set_torque(False)
    start = io.read().positions.copy()
    input("鸭子放在平地上（先扶着），按回车上力矩回 HOME；之后两腿会依次外张/内收几度，手留在旁边别碰…")
    io.set_torque(True)
    rows = []

    def hold(target, n, measure):
        acc = []
        for k in range(n):
            s = io.read()
            io.write(safety.clamp(target))
            if measure and k >= n // 3 and s.imu_ready:
                acc.append((abs(io.currents[LEFT_HIP_ROLL]), abs(io.currents[RIGHT_HIP_ROLL]),
                            math.degrees(math.asin(s.imu.gravity[1]))))
            io.wait_tick()
        return np.mean(acc, axis=0) if acc else None

    try:
        for k in range(100):
            a = min(1.0, k / 100)
            io.read()
            io.write(safety.clamp((1 - a) * start + a * home))
            io.wait_tick()
        hold(home, 75, False)
        prev = home.copy()
        for deg in (0, -6, -4, -2, 0, 2, 4, 6, 0):
            t = home.copy()
            t[LEFT_HIP_ROLL] += math.radians(deg)
            t[RIGHT_HIP_ROLL] -= math.radians(deg)
            for k in range(25):  # 0.5 s glide
                a = (k + 1) / 25
                io.read()
                io.write(safety.clamp((1 - a) * prev + a * t))
                io.wait_tick()
            m = hold(t, 75, True)
            prev = t
            if m is not None:
                rows.append((deg, *m))
                print(f"  spread {deg:+3d} deg: hip roll |I| L {m[0]:5.0f}  R {m[1]:5.0f}  sum {m[0] + m[1]:5.0f} mA   trunk roll {m[2]:+5.1f} deg", flush=True)
        best = min(rows[1:-1], key=lambda r: r[1] + r[2]) if len(rows) > 2 else None
        at_home = [r[1] + r[2] for r in rows if r[0] == 0]
        if best is not None and at_home and best[1] + best[2] > 0.8 * np.mean(at_home):
            best = (0, *best[1:])  # less than 20% better than HOME itself: leave the zeros alone
        if best is not None and best[0] != 0:
            print(f"电流最小在 spread {best[0]:+d} deg（+ = 两腿外张）。输入 y 回车把髋横滚零位按此修正，直接回车放弃：", flush=True)
        else:
            print("HOME 本身电流最小，不用修正。按回车卸力。", flush=True)
        while not select.select([sys.stdin], [], [], 0)[0]:
            io.read()
            io.write(safety.clamp(home))
            io.wait_tick()
        apply = best is not None and best[0] != 0 and sys.stdin.readline().strip().lower() == "y"
    finally:
        io.set_torque(False)
        io.close()
    if apply:
        hip_roll_spread(cal, math.radians(best[0]))  # HOME now stands where the best spread stood
        cal.save(args.calib)
        print(f"hip roll zero_raw L {cal.joints[LEFT_HIP_ROLL].zero_raw}, R {cal.joints[RIGHT_HIP_ROLL].zero_raw}; saved {args.calib}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m duck_runtime.tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("template")
    p.add_argument("out")
    p.add_argument("--layout", choices=["bridge", "official"], default="bridge")
    p.set_defaults(fn=cmd_template)
    for name, fn in [("scan", cmd_scan), ("relax", cmd_relax), ("regs", cmd_regs), ("step-test", cmd_step_test),
                     ("live", cmd_live), ("zero", cmd_zero),
                     ("direction", cmd_direction), ("imu-mount", cmd_imu_mount), ("home-test", cmd_home_test),
                     ("trim", cmd_trim), ("hip-roll", cmd_hip_roll), ("imu-snap", cmd_imu_snap),
                     ("pose-check", cmd_pose_check), ("sign-test", cmd_sign_test), ("mouth", cmd_mouth)]:
        p = sub.add_parser(name)
        add_robot_args(p)
        if name not in ("scan", "relax", "regs", "step-test"):
            p.add_argument("--calib", required=True)
        if name == "step-test":
            p.add_argument("--id", type=int, default=13, help="bridge ID (13 = head_yaw)")
            p.add_argument("--counts", type=int, default=150, help="step size in encoder counts (150 = 13 deg)")
        if name in ("zero", "direction", "pose-check"):
            p.add_argument("--joints", default="", help="comma-separated subset")
        if name == "trim":
            p.add_argument("--target", type=float, default=None,
                           help=f"stance pitch to trim to, deg (+ = nose down); default {HOME_STAND_PITCH_DEG}")
        if name == "mouth":
            p.add_argument("--id", type=int, default=15, help="mouth servo bus ID")
        if name == "home-test":
            p.add_argument("--head-cmd", default="0,0,0,0",
                           help="neck_pitch,head_pitch,head_yaw,head_roll offsets from HOME, degrees")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
