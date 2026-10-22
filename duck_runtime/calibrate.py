"""Zero calibration, one joint at a time, each joint held by its servo once it is right.

    python -m duck_runtime.calibrate --calib calib.json                 # all 14
    python -m duck_runtime.calibrate --calib calib.json --joints legs   # legs | left | right | head | names

Why: with every servo limp, posing joint 5 nudges joints 1-4 that were already right
(and HD1910 backlash lets a limp joint settle anywhere in its play). Here each joint, once
posed, is locked: goal = where it is, torque on that servo only, so it stays put while
you pose the next. A locked joint can be fine-tuned count by count, or released and
re-posed. Nothing is written until you save; the old file is backed up first. Only the
zero (``zero_raw``) of locked joints changes; directions and limits are untouched.

Bridge facts this relies on (MicroDuck_XGO_r1 bridge_runtime.cpp): torque is per servo
(SYNC WRITE 64 rows), any frame from the host refreshes its 1 s watchdog (the loop reads
at 10 Hz), NPD1 P/D writes need every servo off (done once, before the first lock), a
latched fault drops every servo and only an all-14 torque-off clears it.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import select
import shutil
import struct
import sys
import time
import unicodedata

import numpy as np

from .bridge import ADDR_GOAL, BRIDGE_IDS, BridgeError
from .calibration import Calibration

COUNT_DEG = 360.0 / 4096.0
BIG_CHANGE_DEG = 8.0   # a zero that moves more than this is more likely the wrong joint or pose


def _pad(text: str, width: int) -> str:
    """Left-justify by display width (a CJK character takes two columns)."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)

# part, joint, what "zero" looks like
PARTS = {
    "left_hip_yaw": ("左腿", "髋偏航（大腿绕竖直轴转）", "脚尖朝正前方，不内八不外八"),
    "left_hip_roll": ("左腿", "髋横滚（腿向外张/内收）", "腿竖直，与另一条腿平行"),
    "left_hip_pitch": ("左腿", "髋俯仰（大腿前后摆）", "大腿竖直"),
    "left_knee": ("左腿", "膝", "小腿与大腿成一直线"),
    "left_ankle": ("左腿", "踝", "脚底平贴架子底板"),
    "right_hip_yaw": ("右腿", "髋偏航（大腿绕竖直轴转）", "脚尖朝正前方，不内八不外八"),
    "right_hip_roll": ("右腿", "髋横滚（腿向外张/内收）", "腿竖直，与另一条腿平行"),
    "right_hip_pitch": ("右腿", "髋俯仰（大腿前后摆）", "大腿竖直"),
    "right_knee": ("右腿", "膝", "小腿与大腿成一直线"),
    "right_ankle": ("右腿", "踝", "脚底平贴架子底板"),
    "neck_pitch": ("头部", "脖子俯仰", "脖子竖直"),
    "head_pitch": ("头部", "头俯仰（点头）", "脸朝正前方，不抬头不低头"),
    "head_yaw": ("头部", "头偏航（左右转头）", "脸朝正前方"),
    "head_roll": ("头部", "头横滚（歪头）", "头不歪，左右水平"),
}
# Posing order: each leg hip -> foot, then neck -> head.
ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll"]
GROUPS = {"all": ORDER, "legs": ORDER[:10], "left": ORDER[:5], "right": ORDER[5:10], "head": ORDER[10:]}


def label(cal: Calibration, k: int) -> str:
    j = cal.joints[k]
    part, what, _ = PARTS[j.name]
    return f"ID {j.id:2d}  {part} {what}  [{j.name}]"


def resolve_joints(spec: str) -> list[str]:
    names: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        group = GROUPS.get(tok)
        if group is None and tok not in PARTS:
            raise ValueError(f"unknown joint/group {tok!r}: groups {', '.join(GROUPS)} or a joint name")
        names += [n for n in (group or [tok]) if n not in names]
    return [n for n in ORDER if n in names]


class ZeroSession:
    """Lock / nudge / release / measure. No terminal I/O: the console below drives it."""

    def __init__(self, io, joints: list[str], hold_p: int | None = 32, samples: int = 10):
        self.io, self.b, self.cal = io, io.bridge, io.cal
        k_by_name = {j.name: k for k, j in enumerate(self.cal.joints)}
        self.order = [k_by_name[n] for n in joints if self.cal.joints[k_by_name[n]].id]
        self.samples = samples
        self.locked: dict[int, int] = {}          # joint index -> raw goal it is held at
        self.cursor = 0
        self.raw: dict[int, int] = {}
        self.fault = ""
        self._status_t = 0.0
        self.original_pd: dict[int, bytes] = {}
        self.unlock_all()
        time.sleep(0.05)                           # let the bus task apply the torque-off
        if hold_p is not None:
            # NPD1 only with every servo off: set the holding stiffness once, now (D kept).
            for k in self.order:
                i = self.cal.joints[k].id
                pd = self.b.native(i, 50, 2)
                self.original_pd[i] = pd
                if pd[0] != hold_p:
                    self.b.npd(i, hold_p, pd[1])
        self.refresh(force_status=True)

    # -- state ------------------------------------------------------------------------------
    @property
    def current(self) -> int:
        return self.order[self.cursor]

    def id_of(self, k: int) -> int:
        return self.cal.joints[k].id

    def find(self, token: str) -> int:
        """A servo ID or a joint name -> joint index (only joints in this session)."""
        token = token.strip()
        for k in self.order:
            j = self.cal.joints[k]
            if token == j.name or (token.isdigit() and int(token) == j.id):
                return k
        raise KeyError(f"no joint {token!r} in this session")

    def angle_deg(self, k: int) -> float:
        """Angle under the *old* calibration (what the zero pose should read once right)."""
        return math.degrees(self.cal.joints[k].to_rad(self.raw[self.id_of(k)]))

    def refresh(self, force_status: bool = False) -> str:
        """Read every joint (this also feeds the bridge's 1 s watchdog); check for a fault."""
        self.raw = self.io.raw_positions()
        now = time.monotonic()
        if self.locked and (force_status or now - self._status_t > 0.3):
            self._status_t = now
            st = self.b.status()
            mask = sum(1 << (self.id_of(k) - 1) for k in self.locked)
            if st["fault"] or st["torque_mask"] & mask != mask:
                why = st["fault_name"] if st["fault"] else "torque dropped"
                self.fault = f"桥接板故障：{why}，所有关节已断力（输入 a 清除后重新锁定）"
                self.locked.clear()
        return self.fault

    def imu_tilt(self):
        src = getattr(self.io, "imu_source", None)
        if src is None:
            return None
        imu, ready = src.read()
        if not ready:
            return None
        g = imu.gravity
        return (math.degrees(math.asin(max(-1.0, min(1.0, g[0])))),
                math.degrees(math.asin(max(-1.0, min(1.0, g[1])))))

    def _median_raw(self, ids) -> dict[int, int]:
        rows = []
        for _ in range(self.samples):
            rows.append(self.io.raw_positions())
            time.sleep(0.02)
        self.raw = rows[-1]
        return {i: int(round(np.median([r[i] for r in rows]))) for i in ids}

    def _goal(self, i: int, raw: int) -> None:
        d = self.io.diag[i]
        if not d["ft_min"] <= raw <= d["ft_max"]:
            raise BridgeError(f"ID {i}: raw {raw} outside the bridge's limits {d['ft_min']}..{d['ft_max']}")
        self.b.sync_write(ADDR_GOAL, 4, {i: struct.pack("<i", self.io._dx(i, raw))})

    # -- actions ----------------------------------------------------------------------------
    def lock(self, k: int) -> int:
        """Hold joint k where it is now. Returns the raw count it is held at."""
        if self.fault:
            raise BridgeError(self.fault)
        i = self.id_of(k)
        raw = self._median_raw([i])[i]
        self._goal(i, raw)                         # goal first: torque-on never jumps
        self.b.torque([i], True)
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            st = self.b.status()
            if st["fault"]:
                self.locked.clear()
                self.fault = f"桥接板拒绝上力：{st['fault_name']}（输入 a 清除）"
                raise BridgeError(self.fault)
            if st["torque_mask"] >> (i - 1) & 1:
                self.locked[k] = raw
                return raw
            time.sleep(0.01)
        self.b.torque([i], False)
        raise BridgeError(f"ID {i}: torque did not come on within 400 ms")

    def unlock(self, k: int) -> None:
        self.b.torque([self.id_of(k)], False)
        self.locked.pop(k, None)

    def unlock_all(self) -> None:
        """All 14 off: also clears a latched bridge fault."""
        self.b.torque(BRIDGE_IDS, False)
        self.locked.clear()
        self.fault = ""

    def nudge(self, k: int, counts: int) -> int:
        """Move a locked joint by ``counts`` encoder counts in the joint's + direction."""
        if k not in self.locked:
            raise ValueError("先锁定这个关节再微调")
        j = self.cal.joints[k]
        raw = self.locked[k] + counts * j.direction
        self._goal(j.id, raw)
        self.locked[k] = raw
        return raw

    def which_moved(self, seconds: float = 3.0) -> tuple[int, float] | None:
        """The session joint that moved most over ``seconds`` (identify a joint by wiggling it)."""
        start = self.io.raw_positions()
        peak = {k: 0 for k in self.order}
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            now = self.io.raw_positions()
            for k in self.order:
                i = self.id_of(k)
                peak[k] = max(peak[k], abs(now[i] - start[i]))
            time.sleep(0.03)
        self.raw = now
        k = max(peak, key=peak.get)
        return (k, peak[k] * COUNT_DEG) if peak[k] * COUNT_DEG >= 3.0 else None

    def measure(self) -> dict[int, int]:
        """Where each locked joint actually is (median), i.e. its new zero_raw."""
        got = self._median_raw([self.id_of(k) for k in self.locked])
        return {k: got[self.id_of(k)] for k in self.locked}

    def apply(self, zeros: dict[int, int]) -> Calibration:
        new = copy.deepcopy(self.cal)
        for k, raw in zeros.items():
            new.joints[k].zero_raw = int(raw)
        return new

    def close(self) -> None:
        try:
            self.b.torque(BRIDGE_IDS, False)
            if self.original_pd:
                time.sleep(0.05)
                for i, pd in self.original_pd.items():
                    try:
                        self.b.npd(i, pd[0], pd[1])
                    except BridgeError:
                        pass
        finally:
            self.locked.clear()


# -- terminal -----------------------------------------------------------------------------
HELP = """命令（输入后按回车）：
  回车          锁定当前关节（舵机上力保持住）→ 下一个关节
  s             跳过当前关节（不改它的零位）
  b             回到上一个关节
  g <ID/名字>   跳到某个关节，例如 g 8 或 g left_ankle
  u [ID/名字]   松开该关节（断力）重新摆，默认当前关节
  + / -         当前关节（已锁定）微调 1 个计数 = 0.09°；+5、-10 一次调多步
  v             看当前关节 2 秒的实时角度（摆的时候用）
  f             认关节：3 秒内掰动一个关节，告诉你它是哪个 ID / 部位
  l             列出所有关节状态
  w             保存（只写已锁定关节的零位，先自动备份）
  a             全部松开（断力，也清除故障）
  q             退出（全部断力，没保存的不写入）
  h             帮助"""


def _table(s: ZeroSession) -> str:
    lines = [" 序号  ID  部位  " + _pad("关节", 28) + "状态    读数(旧标定)  零位变化"]
    for n, k in enumerate(s.order):
        j = s.cal.joints[k]
        part, what, _ = PARTS[j.name]
        state = "已锁定" if k in s.locked else "松开  "
        mark = "→" if n == s.cursor else " "
        delta = f"{(s.locked[k] - j.zero_raw) * j.direction * COUNT_DEG:+6.2f}°" if k in s.locked else "   -   "
        lines.append(f"{mark}{n + 1:3d}  {j.id:3d}  {part}  {_pad(what, 28)}{state} {s.angle_deg(k):+7.2f}°     {delta}")
    tilt = s.imu_tilt()
    if tilt:
        warn = "   <-- 躯干没放平（>1°），髋零位会把这个倾斜算进去" if max(map(abs, tilt)) > 1.0 else ""
        lines.append(f"躯干：俯仰 {tilt[0]:+.1f}°（+ = 低头） 横滚 {tilt[1]:+.1f}°{warn}")
    return "\n".join(lines)


def _show_current(s: ZeroSession) -> None:
    k = s.current
    _, _, hint = PARTS[s.cal.joints[k].name]
    state = "已锁定" if k in s.locked else "松开，可以摆"
    print(f"\n[{s.cursor + 1}/{len(s.order)}] {label(s.cal, k)}\n"
          f"    零位：{hint}\n    现在 {s.angle_deg(k):+.2f}°（旧标定）  状态：{state}\n> ", end="", flush=True)


def _save(s: ZeroSession, path: str) -> bool:
    if not s.locked:
        print("还没有锁定任何关节，没东西可保存。")
        return False
    zeros = s.measure()
    print("\n要写入的零位（只改已锁定的关节）：")
    for k, raw in zeros.items():
        j = s.cal.joints[k]
        print(f"  {_pad(label(s.cal, k), 56)} zero_raw {j.zero_raw} -> {raw}  "
              f"({(raw - j.zero_raw) * j.direction * COUNT_DEG:+.2f}°)")
    tilt = s.imu_tilt()
    if tilt:
        print(f"  躯干：俯仰 {tilt[0]:+.1f}° 横滚 {tilt[1]:+.1f}°"
              + ("   <-- 超过 1°，建议先放平再存" if max(map(abs, tilt)) > 1.0 else ""))
    if input("写入？输入 y 确认: ").strip().lower() != "y":
        print("没有写入。")
        return False
    if os.path.exists(path):
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        print(f"旧文件备份到 {backup}")
    s.apply(zeros).save(path)
    s.cal = Calibration.load(path)
    s.io.cal = s.cal
    print(f"已保存 {path}。关节仍保持锁定；q 退出。")
    return True


def _read_line(timeout: float) -> str | None:
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def console(s: ZeroSession, path: str) -> None:
    print(__doc__.split("\n\n")[1].strip() + "\n\n" + HELP)
    print("\n把鸭子放进零位架子。每个关节：摆好 → 回车锁定。\n")
    print(_table(s))
    _show_current(s)
    saved, shown_fault = True, ""
    while True:
        try:
            msg = s.refresh()
            if msg and msg != shown_fault:
                print(f"\n!! {msg}\n> ", end="", flush=True)
            shown_fault = msg
            line = _read_line(0.1)
        except EOFError:
            break
        if line is None:
            continue
        cmd, _, arg = line.strip().partition(" ")
        try:
            if cmd == "":
                k = s.current
                if k not in s.locked:
                    raw = s.lock(k)
                    saved = False
                    d = (raw - s.cal.joints[k].zero_raw) * s.cal.joints[k].direction * COUNT_DEG
                    print(f"  锁定 ID {s.id_of(k)}，零位变化 {d:+.2f}°"
                          + (f"   <-- 变化超过 {BIG_CHANGE_DEG:.0f}°，确认摆的是这个关节、姿势对吗？不对就 u 松开重摆"
                             if abs(d) > BIG_CHANGE_DEG else ""))
                if s.cursor < len(s.order) - 1:
                    s.cursor += 1
                else:
                    print("  最后一个关节了：l 检查，w 保存。")
            elif cmd == "s":
                s.cursor = min(s.cursor + 1, len(s.order) - 1)
            elif cmd == "b":
                s.cursor = max(s.cursor - 1, 0)
            elif cmd == "g":
                s.cursor = s.order.index(s.find(arg))
            elif cmd == "u":
                k = s.find(arg) if arg else s.current
                s.unlock(k)
                s.cursor = s.order.index(k)
                print(f"  已松开 ID {s.id_of(k)}，摆好后回车重新锁定")
            elif cmd and set(cmd) <= {"+", "-"} or (cmd[:1] in "+-" and cmd[1:].isdigit()):
                n = int(cmd) if cmd[1:].isdigit() else (cmd.count("+") - cmd.count("-"))
                raw = s.nudge(s.current, n)
                saved = False
                time.sleep(0.15)
                s.refresh()
                print(f"  ID {s.id_of(s.current)} 目标 raw {raw}，现在 {s.angle_deg(s.current):+.2f}°")
            elif cmd == "v":
                for _ in range(10):
                    s.refresh()
                    print(f"\r  {label(s.cal, s.current)}  {s.angle_deg(s.current):+7.2f}°   ", end="", flush=True)
                    time.sleep(0.2)
                print()
            elif cmd == "f":
                print("  3 秒内来回掰动你想找的那个关节（松开的关节才动得了）…")
                got = s.which_moved()
                if got is None:
                    print("  没检测到明显转动（< 3°）")
                else:
                    k, deg = got
                    print(f"  动的是：{label(s.cal, k)}（{deg:.0f}°）  g {s.id_of(k)} 可以跳过去")
            elif cmd == "l":
                print(_table(s))
            elif cmd == "w":
                saved = _save(s, path) or saved
            elif cmd == "a":
                s.unlock_all()
                print("  全部松开")
            elif cmd == "q":
                if not saved and s.locked and input("有锁定的关节还没保存，确定退出？(y): ").strip().lower() != "y":
                    pass
                else:
                    break
            elif cmd == "h":
                print(HELP)
            else:
                print("  不认识的命令，h 看帮助")
        except (KeyError, ValueError, BridgeError) as e:
            print(f"  {e}")
        _show_current(s)


def main(argv=None):
    from .tools import add_robot_args, open_io
    ap = argparse.ArgumentParser(prog="python -m duck_runtime.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_robot_args(ap)
    ap.add_argument("--calib", required=True, help="calibration file to read (and write, unless --out)")
    ap.add_argument("--out", help="write here instead of --calib")
    ap.add_argument("--joints", default="all", help="all | legs | left | right | head | comma-separated names")
    ap.add_argument("--hold-p", type=int, default=32,
                    help="RAM P gain while holding (restored on exit); 0 = leave the servos' P as is")
    args = ap.parse_args(argv)
    if args.backend != "bridge":
        ap.error("only the bridge backend supports per-joint holding")
    joints = resolve_joints(args.joints)
    cal = Calibration.load(args.calib)
    io = open_io(args, cal)
    s = None
    try:
        s = ZeroSession(io, joints, hold_p=args.hold_p or None)
        console(s, args.out or args.calib)
    except KeyboardInterrupt:
        print()
    finally:
        if s is not None:
            s.close()
        io.close()
        print("全部断力，已退出。")


if __name__ == "__main__":
    main(sys.argv[1:])
