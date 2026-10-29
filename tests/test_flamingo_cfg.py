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
    for name in ("flamingo_composite", "flamingo_contact", "flamingo_swing_lift", "flamingo_com",
                 "flamingo_upright", "flamingo_pose", "flamingo_stillness"):
        assert r[name].weight > 0, name
    # mjlab-style costs (>= 0) → negative weight
    for name in ("non_sole_contact", "hip_hard_limits", "dof_pos_limits", "action_rate_l2",
                 "self_collisions", "body_ang_vel", "angular_momentum"):
        assert r[name].weight < 0, name
    # self-negating microduck penalty (returns -|a_z|) → POSITIVE weight
    assert r["gentle_motion"].weight > 0


def test_hip_joints_exempt_from_soft_limits(cfg, model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    import re
    pat = cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names[0]
    covered = [n for n in names if n and re.fullmatch(pat, n)]
    assert not any("hip_roll" in n or "hip_yaw" in n for n in covered)
    assert any("knee" in n for n in covered)


@pytest.mark.parametrize("pose", [fl.POSE_RIGHT, fl.POSE_LEFT])
def test_pose_clear_of_hard_stops(pose, model):
    """The target itself must not sit in the hip_hard_limits band."""
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
