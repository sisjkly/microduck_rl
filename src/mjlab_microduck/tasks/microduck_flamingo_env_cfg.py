"""Microduck FLAMINGO — commanded single-leg stand ↔ two feet, either leg, one policy.

Command (twist slot, community flamingo-cycle convention, tests/test_publish_manifest.py):
    twist = [flag, side, 0]   flag 0 = two feet, 1 = one foot
                              side +1 = RIGHT foot down (left lifted), -1 = LEFT foot down
    [0, 0, 0] = two feet = the deployment idle every policy shares.
61D obs unchanged (head command live, body command zero-padded) → hot-swappable.

Built ON the sitstand recipe (make_microduck_sitstand_env_cfg): its DR / obs noise /
delays / encoder bias / IMU misalignment / contact-solver hardening / delayed pushes /
smoothness curricula come in by construction (AGENTS.md: build on the closest template;
commanded two-state → sitstand). The sit-specific rewards, events and curricula are
removed and replaced by the flamingo stack below.

Physics facts this cfg is built on (CPU MuJoCo + training BAM M6 actuator, 2026-10-28):
  - Open loop, the robot cannot even stand on two feet: effective servo stiffness
    ≈ 0.55 N·m/rad vs ≈ 0.8 needed (16/16 zero-action envs fall within 0.5 s). Every
    stance here is CLOSED-LOOP; static targets are only reward references.
  - No ankle roll + hip-roll axis 24 mm inboard of the sole + sole half-width 17.8 mm
    ⇒ the CoM only enters a single sole with ≥ ~15° of trunk lean toward the support
    side (head used as counterweight) / ~16-20° with the head fixed. At 25° lean the
    solved swan pose below keeps its CoM 14.8 mm inside the sole edge; the support
    hip_roll then sits at its 0.9 soft limit (-0.345).
  - Static holding torques ≤ 0.27 N·m vs a 0.78-0.96 N·m budget, BUT the soft servos
    sag 0.3-0.5 rad under that load: the policy must command offsets (gravity
    feed-forward). Action scale 1.0 + wide ctrlrange allow it.
  - With both soles flat the CoM cannot enter one sole (best -8.5 mm with the head,
    -14 mm without): lift-off is DYNAMIC (a weight shift carried by momentum). Two
    scripted joint-space transitions failed 0/24 — the transition is what RL has to
    discover; the rewards must not prescribe its path.
  - A hand-written 50 Hz / 25 ms-delay controller holds the pose 5 s in 5/6 noisy
    starts at 15 mm margin, 1/6 at 10 mm (pose then had a trunk contact, since fixed): margin is everything → reward the CoM over
    the sole centre directly, not just "a foot in the air".
  - The model is L/R symmetric (mirrored pose: 15.0 vs 15.1 mm margin).

Reward design (AGENTS.md rules applied):
  - Slewed target (FlamingoCommand.alpha, 1 s): rewards track a moving blend between
    the two-feet target (HOME, upright, both soles down) and the one-foot target
    (solved pose, commanded lean, CoM over the support sole, swing foot up) — no
    jackpot for arriving early, the path is free.
  - Multiplicative composite (contact × upright × pose × CoM): a lean-without-lift or
    lift-without-balance compromise collapses on its weakest factor.
  - Contact pattern is binary and gated on not being toppled; the swing-lift and
    swing-behind terms only pay while the support sole is down → hopping and lying both
    lose. swing-behind fixes the lift DIRECTION (backward, swan style).
  - Every non-sole ground contact costs; falls terminate past 50° (target lean 25°).
  - Positive task mass ≈ sitstand's, so its sim2real regularisers keep their RELATIVE
    strength. Penalty signs: mjlab costs (≥ 0) negative weight; self-negating
    functions positive weight — check Episode_Reward/<penalty> ≤ 0 on every run.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

NUM_STEPS_PER_ENV = 24

# Mirror loss: the task is L/R symmetric by construction (side lives in the vy slot,
# which symmetry.py negates). OFF like every other env until an A/B run validates it
# on the critic layout of this env.
ENABLE_SYMMETRY = False

# ── Command ───────────────────────────────────────────────────────────────────
EPISODE_LENGTH_S = 12.0
DWELL_S = (3.0, 6.0)          # time in each commanded state before a resample may flip it
ONE_FOOT_PROB = 0.5           # P(resample commands one foot) — all four (start × command) cases
RAMP_S = 1.0                  # alpha 0 ↔ 1 slew time (reward reference only; obs = raw flag)

# ── Solved one-foot pose (RIGHT foot down), servo order 0-4 L leg, 5-8 head, 9-13 R leg.
# SWAN / attitude style (v2, 2026-10-29): the swing leg is lifted BACKWARD — thigh back,
# knee folded (-1.33 rad), foot 66 mm behind the support foot along the trunk heading and
# 118 mm off the floor (lowest sole point; site 153 mm). pose_solve.py with BEHIND=0.04,
# MIN_LIFT=0.04, MAX_LIFT=0.12, MAX_TILT=25, hip_yaw + head free: support sole flat, CoM
# 14.8 mm inside the sole edge, trunk lean 24.9°, no self-contact (swing leg 25 mm from
# the support leg, 4.4 mm from the trunk), every hip joint ≥ 0.0386 rad from its hard
# stop, head used as counterweight (head_yaw -1.48). LEFT = L/R mirror.
# v1 lifted the leg FORWARD (foot 43 mm in front, margin 13.2 mm) — trained fine but was
# not the intended motion. The hand-written probe controller holds v1 6/6 but this pose
# only 1-2/6 with the same (stance-tuned) ankle gains: a pitch oscillation, the CoM stays
# 9-13 mm inside the sole — a probe-gain issue, see the gain sweep in the physics notes.
POSE_RIGHT = (
    -0.1217, 0.3431, 0.6277, -1.3316, 0.5811,      # left leg (swing)
    0.6771, 0.0556, -1.4792, -0.374,               # neck_pitch, head_pitch, head_yaw, head_roll
    0.3872, -0.3454, 0.5814, 0.7742, 0.2404,       # right leg (support)
)


def mirror_pose(p):
    """L/R mirror (symmetry.py rule): swap legs and negate every leg joint; head_yaw and
    head_roll negate, neck/head pitch stay."""
    p = list(p)
    return tuple([-v for v in p[9:14]] + [p[5], p[6], -p[7], -p[8]] + [-v for v in p[0:5]])


POSE_LEFT = mirror_pose(POSE_RIGHT)
# Trunk state of POSE_RIGHT with the support sole flat on the floor (harness.place_on_floor).
POSE_TRUNK_Z = 0.12647
POSE_QUAT_RIGHT = (0.95876, 0.21511, -0.01812, 0.18487)
GRAV_B_RIGHT = (-0.1143, -0.4058, -0.9068)     # projected gravity, body frame, at POSE_RIGHT

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward constants ──────────────────────────────────────────────────────────
SWING_LIFT_TARGET = 0.08      # m, foot-site height (pose: 0.153 site / 0.118 lowest sole point)
SWING_BEHIND_TARGET = 0.05    # m, swing foot behind the support foot along the trunk heading (pose: 0.066)
COM_STD = 0.015               # m — ≈ the sole half-width: the margin that decides holds
UPRIGHT_STD = 0.25            # |Δg| ≈ 14° — wide enough that the current policy scores
POSE_STD = 0.5                # rad, legs (HOME → pose deltas up to ~0.96 rad)
FALL_TILT_DEG = 50.0          # termination (target lean 25°)
LIMIT_MARGIN = 0.03           # rad from the HARD stop for hip_roll / hip_yaw

# Spawn curriculum: many starts already on one foot while holding is learned, then more
# HOME starts so lift-off gets the data (phase-align: don't harden before it consolidates).
SPAWN_STAGES = [
    {"step": 0,                        "params": {"prob": 0.5}},
    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prob": 0.3}},
]


def make_microduck_flamingo_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    assert not rough, "flamingo is flat-only for now (foot heights are measured vs the env origin)"
    cfg = make_microduck_sitstand_env_cfg(play=play, rough=False)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Strip the sit-specific stack ──────────────────────────────────────────
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_height", "posture_height_sharp",
                 "posture_height_l1", "rise_bootstrap", "descent_speed", "rise_speed",
                 "upright_linear", "upright_while_tall", "posture_stillness", "posture_composite",
                 "neck_strain_leaned"):
        cfg.rewards.pop(name, None)
    for name in ("set_ground_state", "seated_head_jerk", "seated_tip"):
        cfg.events.pop(name, None)
    for name in ("descent_speed_weight", "rise_speed_weight", "seated_head_jerk_prob",
                 "seated_tip_range", "head_pose_range"):
        cfg.curriculum.pop(name, None)

    # ── Sensors: non-sole ground contact ──────────────────────────────────────
    # Every robot body except the two ankle bodies (each carries only its sole
    # collision geom on the groundcontact model) vs the terrain.
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^(?!ankle_left$|ankle_right$).*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (body_ground_cfg,)
    feet = "feet_ground_contact"   # sitstand's sensor: LEFT sole first, RIGHT second

    # ── Command: [flag, side, 0] with a slewed target ─────────────────────────
    base = cfg.commands["twist"]
    base_fields = {f: getattr(base, f) for f in UniformVelocityCommandCfg.__dataclass_fields__}
    base_fields.update(rel_standing_envs=0.0, rel_heading_envs=0.0, heading_command=False,
                       resampling_time_range=DWELL_S, debug_vis=False)
    base_fields["ranges"].heading = None
    cfg.commands["twist"] = microduck_mdp.FlamingoCommandCfg(
        **base_fields, one_foot_prob=ONE_FOOT_PROB, ramp_s=RAMP_S,
    )

    # ── Task rewards (positive mass ≈ sitstand's) ─────────────────────────────
    common = {"command_name": "twist"}
    cfg.rewards["flamingo_composite"] = RewardTermCfg(
        func=microduck_mdp.flamingo_composite, weight=4.0,
        params={**common, "sensor_name": feet, "pose_right": POSE_RIGHT, "pose_left": POSE_LEFT,
                "joint_indices": _LEG_JOINTS, "grav_right": GRAV_B_RIGHT,
                "pose_std": POSE_STD, "upright_std": UPRIGHT_STD, "com_std": 2 * COM_STD},
    )
    cfg.rewards["flamingo_contact"] = RewardTermCfg(
        func=microduck_mdp.flamingo_contact_pattern, weight=2.0,
        params={**common, "sensor_name": feet, "gate_tilt_deg": 45.0},
    )
    cfg.rewards["flamingo_swing_lift"] = RewardTermCfg(
        func=microduck_mdp.flamingo_swing_lift, weight=1.5,
        params={**common, "sensor_name": feet, "target_height": SWING_LIFT_TARGET, "gate_tilt_deg": 45.0},
    )
    # Lift DIRECTION: backward (swan). The leg-pose Gaussian is too loose to pin it alone.
    cfg.rewards["flamingo_swing_behind"] = RewardTermCfg(
        func=microduck_mdp.flamingo_swing_behind, weight=1.5,
        params={**common, "sensor_name": feet, "target_dist": SWING_BEHIND_TARGET, "gate_tilt_deg": 45.0},
    )
    cfg.rewards["flamingo_com"] = RewardTermCfg(
        func=microduck_mdp.flamingo_com_over_support, weight=2.0,
        params={**common, "std": COM_STD},
    )
    cfg.rewards["flamingo_upright"] = RewardTermCfg(
        func=microduck_mdp.flamingo_upright, weight=2.0,
        params={**common, "grav_right": GRAV_B_RIGHT, "std": UPRIGHT_STD},
    )
    cfg.rewards["flamingo_pose"] = RewardTermCfg(
        func=microduck_mdp.flamingo_pose, weight=1.0,
        params={**common, "pose_right": POSE_RIGHT, "pose_left": POSE_LEFT,
                "joint_indices": _LEG_JOINTS, "std": POSE_STD},
    )
    cfg.rewards["flamingo_stillness"] = RewardTermCfg(
        func=microduck_mdp.flamingo_hold_stillness, weight=1.0,
        params={**common, "sensor_name": feet, "ang_vel_std": 0.5},
    )
    # Head: commandable but LIGHT — it is 38 % of the mass and the best counterweight
    # (≈ +5 mm margin); the head command keeps sitstand's stage-0 alive ranges only.
    cfg.rewards["head_pose_tracking"].weight = 0.3

    # ── Costs ─────────────────────────────────────────────────────────────────
    cfg.rewards["non_sole_contact"] = RewardTermCfg(
        func=microduck_mdp.flamingo_non_sole_contact, weight=-2.0,
        params={"sensor_name": body_ground_cfg.name},
    )
    # The one-foot pose needs hip_roll / hip_yaw AT the 0.9 soft limit: exempt them from
    # the soft-limit cost and guard the HARD stops instead.
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*hip_roll|.*hip_yaw).*",))
    cfg.rewards["hip_hard_limits"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll", r".*hip_yaw")),
                "margin": LIMIT_MARGIN},
    )
    # Kept from sitstand (sim2real parity): action_rate_l2 (+ its -0.1 → -1.0 ramp),
    # joint_torque_rate_l2 (+ ramp), gentle_motion |a_z| (POSITIVE weight, self-negating),
    # body_ang_vel, angular_momentum, self_collisions, head_pose_tracking.

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(FALL_TILT_DEG)},
    )

    # ── Events: reverse-curriculum one-foot spawns (after the base HOME reset) ─
    cfg.events["flamingo_spawn"] = EventTermCfg(
        func=microduck_mdp.flamingo_spawn, mode="reset",
        params={"prob": 0.5 if play else SPAWN_STAGES[0]["params"]["prob"],
                "pose_right": POSE_RIGHT, "pose_left": POSE_LEFT,
                "trunk_z": POSE_TRUNK_Z, "quat_right": POSE_QUAT_RIGHT, "joint_noise_std": 0.01},
    )
    if not play:
        cfg.curriculum["flamingo_spawn_prob"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={"event_name": "flamingo_spawn", "param_stages": SPAWN_STAGES},
        )

    return cfg


MicroduckFlamingoRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_flamingo",
    run_name="microduck_flamingo",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6_000,
)
