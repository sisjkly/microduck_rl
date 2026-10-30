"""Cfg invariants for the flamingo (single-leg stand) task. CPU only, never steps an env."""
import math

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_flamingo_env_cfg as fl
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg

TASKS = ("Mjlab-Flamingo-Flat-MicroDuck", "Mjlab-Flamingo-Flat-Backlash-MicroDuck")


@pytest.fixture(scope="module")
def cfg():
    return fl.make_microduck_flamingo_env_cfg()


@pytest.fixture(scope="module")
def model():
    return MICRODUCK_STANDUP_ROBOT_CFG.spec_fn().compile()


def test_registered():
    assert set(TASKS) <= set(list_tasks())


def test_command_contract(cfg):
    tw = cfg.commands["twist"]
    assert isinstance(tw, microduck_mdp.FlamingoCommandCfg)
    assert tw.heading_command is False and tw.ranges.heading is None
    assert tw.rel_standing_envs == 0.0
    assert 0.0 < tw.one_foot_prob < 1.0          # both holds and both transitions trained
    assert tw.ramp_s > 0.0


def test_obs_layout_matches_sitstand(cfg):
    ref = make_microduck_sitstand_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(ref.observations["actor"].terms)


def test_sit_stack_removed(cfg):
    for name in ("posture_composite", "posture_height_l1", "descent_speed", "rise_speed",
                 "upright_linear", "neck_strain_leaned"):
        assert name not in cfg.rewards
    for name in ("set_ground_state", "seated_tip", "seated_head_jerk"):
        assert name not in cfg.events


def test_reward_signs(cfg):
    r = cfg.rewards
    for name in ("flamingo_composite", "flamingo_contact", "flamingo_swing_lift", "flamingo_swing_behind", "flamingo_com",
                 "flamingo_upright", "flamingo_pose", "flamingo_stillness"):
        assert r[name].weight > 0, name
    # mjlab-style costs (>= 0) → negative weight
    for name in ("non_sole_contact", "joint_limits", "action_over_limit", "servo_stall", "head_range",
                 "dof_pos_limits", "action_rate_l2", "self_collisions", "body_ang_vel", "angular_momentum"):
        assert r[name].weight < 0, name
    # self-negating microduck penalty (returns -|a_z|) → POSITIVE weight
    assert r["gentle_motion"].weight > 0


def test_limit_costs_cover_every_servo(cfg, model):
    """v2 lesson: jammed joints were outside every limit term. Now the hard-stop band and the
    soft-limit cost cover ALL servo joints (head included), passive joints excluded."""
    import re
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    servos = [n for n in names if n and not n.startswith("passive_") and "freejoint" not in n]
    pat = cfg.rewards["joint_limits"].params["asset_cfg"].joint_names[0]
    assert all(re.fullmatch(pat, n) for n in servos)
    assert 0.04 <= cfg.rewards["joint_limits"].params["margin"] < 0.08   # target sits 0.08 from every stop
    assert "asset_cfg" not in cfg.rewards["dof_pos_limits"].params or \
        cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names in ((".*",), None)


def test_pose_respects_soft_limits_and_head_caps(model):
    """The goal state must pay nothing in dof_pos_limits (0.9 soft) and head_range."""
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    import re
    for pose in (fl.POSE_RIGHT, fl.POSE_LEFT):
        for j, v in zip(_JOINTS, pose):
            lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            assert mid - 0.9 * half <= v <= mid + 0.9 * half, (j, v)
        for k, cap in zip(range(5, 9), fl.HEAD_CAPS):
            home = next(val for pat, val in HOME_FRAME.joint_pos.items() if re.search(pat, _JOINTS[k]))
            assert abs(pose[k] - home) <= cap + 1e-3, (_JOINTS[k], pose[k], home, cap)


def test_pushes_halved(cfg):
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert max(abs(v) for st in stages for rng in st["velocity_range"].values() for v in rng) <= 0.15 + 1e-9


@pytest.mark.parametrize("pose", [fl.POSE_RIGHT, fl.POSE_LEFT])
def test_pose_clear_of_hard_stops(pose, model):
    """The target itself must not sit in the joint_limits band."""
    joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "neck_pitch", "head_pitch", "head_yaw", "head_roll",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
    for j, v in zip(joints, pose):
        lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        assert lo + fl.LIMIT_MARGIN < v < hi - fl.LIMIT_MARGIN, (j, v, lo, hi)


def test_mirror_is_involution_and_symmetric():
    assert np.allclose(fl.mirror_pose(fl.mirror_pose(fl.POSE_RIGHT)), fl.POSE_RIGHT)
    assert np.allclose(fl.POSE_LEFT, fl.mirror_pose(fl.POSE_RIGHT))


def test_pose_constants_consistent():
    q = np.array(fl.POSE_QUAT_RIGHT)
    assert abs(np.linalg.norm(q) - 1) < 1e-3
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    g_b = R.reshape(3, 3).T @ np.array([0, 0, -1.0])
    assert np.allclose(g_b, fl.GRAV_B_RIGHT, atol=2e-3)
    lean = math.degrees(math.acos(-g_b[2]))
    assert lean < fl.FALL_TILT_DEG - 15        # room to wobble before termination
    assert g_b[1] < 0                          # RIGHT support leans toward the right (body -y)


def test_spawn_and_curriculum(cfg):
    ev = cfg.events["flamingo_spawn"]
    assert ev.mode == "reset" and 0 < ev.params["prob"] <= 1
    order = list(cfg.events)
    assert order.index("flamingo_spawn") > order.index("reset_base")   # overrides the HOME reset
    stages = cfg.curriculum["flamingo_spawn_prob"].params["param_stages"]
    assert [s["step"] for s in stages] == sorted(s["step"] for s in stages)


def test_backlash_variant_model():
    m = load_env_cfg(TASKS[1]).scene.entities["robot"].spec_fn().compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert sum(n.startswith("passive_") and n.endswith("_backlash") for n in names) == 14


_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
           "neck_pitch", "head_pitch", "head_yaw", "head_roll",
           "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_target_pose_lifts_leg_backward(side, model):
    """Swan style: the target's swing foot is BEHIND the support foot (along the trunk
    heading) by at least the reward's target distance, and well off the floor."""
    d = mujoco.MjData(model)
    pose = fl.POSE_RIGHT if side == "right" else fl.POSE_LEFT
    quat = np.array(fl.POSE_QUAT_RIGHT)
    if side == "left":
        quat[1], quat[3] = -quat[1], -quat[3]
    d.qpos[:3] = (0, 0, fl.POSE_TRUNK_Z)
    d.qpos[3:7] = quat / np.linalg.norm(quat)
    for j, v in zip(_JOINTS, pose):
        d.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_kinematics(model, d)
    site = lambda n: d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)]
    swing, sup = ("left", "right") if side == "right" else ("right", "left")
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    fwd = d.xmat[tb].reshape(3, 3)[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    behind = -np.dot(site(f"{swing}_foot") - site(f"{sup}_foot"), fwd)
    assert behind >= fl.SWING_BEHIND_TARGET, behind
    assert site(f"{swing}_foot")[2] - site(f"{sup}_foot")[2] >= fl.SWING_LIFT_TARGET
