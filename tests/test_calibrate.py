"""Per-joint lock-and-hold zero calibration, against the emulated MDK2 firmware."""

import io as _io
import os
import sys

import pytest

from duck_runtime import calibrate as C
from duck_runtime.bridge import BridgeError
from duck_runtime.calibration import Calibration
from test_bridge import _eventually, rig  # noqa: F401  (fixture)


def bit(i):
    return 1 << (i - 1)


def test_resolve_joints_groups_names_and_order():
    assert C.resolve_joints("legs") == C.ORDER[:10]
    assert C.resolve_joints("head,left_ankle") == ["left_ankle"] + C.ORDER[10:]
    assert C.resolve_joints("right_knee,right_hip_yaw") == ["right_hip_yaw", "right_knee"]
    with pytest.raises(ValueError):
        C.resolve_joints("left_foot")
    assert set(C.PARTS) == set(C.ORDER) and "mouth" not in C.PARTS


def test_label_has_id_part_and_name(rig):
    fake, io, cal = rig
    k = [j.name for j in cal.joints].index("left_ankle")
    assert C.label(cal, k) == f"ID {cal.joints[k].id:2d}  左腿 踝  [left_ankle]"


def test_lock_holds_only_that_servo_without_a_jump(rig):
    fake, io, cal = rig
    s = C.ZeroSession(io, C.ORDER, hold_p=20, samples=3)
    assert all(fake.pd[cal.joints[k].id][0] == 20 for k in s.order)       # holding stiffness, set while off
    a, b = s.order[0], s.order[1]
    ia, ib = cal.joints[a].id, cal.joints[b].id
    before = dict(fake.raw)
    assert s.lock(a) == before[ia]
    assert fake.torque == bit(ia) and fake.raw == before                  # goal = present: nothing moved
    s.lock(b)
    assert fake.torque == bit(ia) | bit(ib)
    s.unlock(a)
    assert _eventually(lambda: fake.torque == bit(ib)) and a not in s.locked
    s.close()
    assert _eventually(lambda: fake.torque == 0) and all(fake.pd[cal.joints[k].id] == [32, 40] for k in s.order)  # restored


def test_nudge_moves_in_the_joint_plus_direction(rig):
    fake, io, cal = rig
    s = C.ZeroSession(io, ["left_ankle"], samples=3)
    k = s.current
    s.lock(k)
    s.refresh()
    a0 = s.angle_deg(k)
    s.nudge(k, +5)
    s.refresh()
    assert s.angle_deg(k) - a0 == pytest.approx(5 * C.COUNT_DEG)
    s.unlock(k)
    with pytest.raises(ValueError):
        s.nudge(k, 1)


def test_goal_outside_bridge_limits_is_refused_not_faulted(rig):
    fake, io, cal = rig
    s = C.ZeroSession(io, ["left_knee"], samples=3)
    k = s.current
    i = cal.joints[k].id
    fake.raw[i] = 4094
    s.lock(k)
    with pytest.raises(BridgeError):
        s.nudge(k, 5 * cal.joints[k].direction)
    assert fake.fault == 0 and fake.torque == bit(i)


def test_fault_drops_every_lock_and_a_clears_it(rig):
    fake, io, cal = rig
    s = C.ZeroSession(io, C.ORDER, samples=3)
    s.lock(s.order[0]); s.lock(s.order[1])
    fake.fault, fake.torque = 1, 0                                         # host timeout, say
    assert "故障" in s.refresh(force_status=True) and not s.locked
    with pytest.raises(BridgeError):
        s.lock(s.order[0])
    s.unlock_all()
    assert _eventually(lambda: fake.fault == 0) and s.fault == ""
    s.lock(s.order[0])
    assert fake.torque == bit(cal.joints[s.order[0]].id)


def test_measure_and_apply_change_only_locked_zeros(rig):
    fake, io, cal = rig
    s = C.ZeroSession(io, C.ORDER, samples=3)
    a = s.order[3]
    s.lock(a)
    s.nudge(a, -7)
    new = s.apply(s.measure())
    for k, (old, nj) in enumerate(zip(cal.joints, new.joints)):
        assert nj.direction == old.direction and nj.min_rad == old.min_rad
        if k == a:
            assert nj.zero_raw == fake.raw[old.id]
        else:
            assert nj.zero_raw == old.zero_raw


def test_console_lock_nudge_save_with_backup(rig, tmp_path, monkeypatch):
    fake, io, cal = rig
    path = tmp_path / "calib.json"
    cal.save(path)
    s = C.ZeroSession(io, ["left_hip_yaw", "left_hip_roll"], samples=3)
    r, w = os.pipe()
    os.write(w, "\n+3\ng left_hip_yaw\nl\nw\ny\nq\n".encode())
    os.close(w)
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    monkeypatch.setattr(sys, "stdout", _io.StringIO())
    C.console(s, str(path))
    out = sys.stdout.getvalue()
    monkeypatch.undo()
    assert "已保存" in out
    saved = Calibration.load(path)
    k0, k1 = s.order
    i0, i1 = cal.joints[k0].id, cal.joints[k1].id
    assert saved.joints[k0].zero_raw == fake.raw[i0]           # locked by Enter
    assert saved.joints[k1].zero_raw == cal.joints[k1].zero_raw  # +3 hit an unlocked joint: refused, untouched
    assert len(list(tmp_path.glob("calib.json.bak-*"))) == 1
    s.close()


def test_which_moved_names_the_wiggled_joint(rig):
    import threading
    fake, io, cal = rig
    s = C.ZeroSession(io, C.ORDER, samples=3)
    k = [j.name for j in cal.joints].index("right_hip_roll")
    i = cal.joints[k].id
    threading.Timer(0.2, lambda: fake.raw.__setitem__(i, fake.raw[i] + 120)).start()
    got = s.which_moved(seconds=0.6)
    assert got is not None and got[0] == k and got[1] == pytest.approx(120 * C.COUNT_DEG)
    assert s.which_moved(seconds=0.2) is None                  # nothing moved
