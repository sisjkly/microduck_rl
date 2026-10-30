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
    solved swan pose below keeps its CoM 10.6 mm inside the sole edge with every joint
    >= 0.08 rad from its hard stop and the head bounded (14.8 mm when the hips and head
    were allowed onto their stops — v2, which the policy then exploited by jamming them).
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
    swing-behind terms only pay while the support sole is down AND the swing sole is off
    the floor → hopping, lying and the toe-planted kickstand (v3) all lose. swing-behind
    fixes the lift DIRECTION (backward, swan style); trunk height prices a crouch.
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
# SWAN / attitude style, v3 (2026-10-30): swing leg lifted BACKWARD (thigh back, knee
# folded -1.25 rad), foot 73 mm behind the support foot along the trunk heading and 119 mm
# off the floor (lowest sole point; site 155 mm). pose_solve.py with BEHIND=0.04,
# MIN_LIFT=0.04, MAX_LIFT=0.12, MAX_TILT=25, STOP_MARGIN=0.08, HEAD_CAPS=neck_pitch:0.6,
# head_pitch:0.6, head_yaw:0.5, head_roll:0.25: support sole flat, CoM 10.6 mm inside the
# sole edge, trunk lean 24.6°, no self-contact, EVERY joint >= 0.080 rad from its hard
# stop, head only a bounded counterweight (yaw -0.50, roll -0.20 from HOME). LEFT = mirror.
# History: v1 lifted the leg FORWARD; v2 (backward, head free, hips 0.0386 rad from the
# stops) trained to 32/32 hold / lift-off / put-down in sim, but its policy jammed five
# joints into their hard stops with commands far past them (head_roll +7.05 rad commanded
# vs a 0.44 limit, support hip_yaw +3.38, hip_roll -2.34) and twisted the head 116° as a
# counterweight — a stalled-servo trick that would not survive the real robot. v3 moves
# the target off the stops and bounds the head; the costs below price the jam itself.
POSE_RIGHT = (
    -0.1731, 0.2983, 0.5275, -1.2462, 0.4609,      # left leg (swing)
    0.776, -0.1638, -0.4994, -0.197,               # neck_pitch, head_pitch, head_yaw, head_roll
    0.3563, -0.3032, 0.7664, 0.7285, 0.1442,       # right leg (support)
)


def mirror_pose(p):
    """L/R mirror (symmetry.py rule): swap legs and negate every leg joint; head_yaw and
    head_roll negate, neck/head pitch stay."""
    p = list(p)
    return tuple([-v for v in p[9:14]] + [p[5], p[6], -p[7], -p[8]] + [-v for v in p[0:5]])


POSE_LEFT = mirror_pose(POSE_RIGHT)
# Trunk state of POSE_RIGHT with the support sole flat on the floor (harness.place_on_floor).
POSE_TRUNK_Z = 0.12903
POSE_QUAT_RIGHT = (0.96321, 0.20615, 0.05476, 0.16348)
GRAV_B_RIGHT = (0.0381, -0.415, -0.909)        # projected gravity, body frame, at POSE_RIGHT

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward constants ──────────────────────────────────────────────────────────
SWING_LIFT_TARGET = 0.08      # m, foot-site height (pose: 0.155 site / 0.119 lowest sole point)
SWING_BEHIND_TARGET = 0.05    # m, swing foot behind the support foot along the trunk heading (pose: 0.073)
COM_STD = 0.015               # m — ≈ the sole half-width: the margin that decides holds
UPRIGHT_STD = 0.25            # |Δg| ≈ 14° — wide enough that the current policy scores
POSE_STD = 0.5                # rad, legs (HOME → pose deltas up to ~0.96 rad)
FALL_TILT_DEG = 50.0          # termination (target lean 25°)
LIMIT_MARGIN = 0.05           # rad from the HARD stop, ALL servo joints (v2: 0.03, hips only). The target
                              # sits 0.08 from every stop: 0.03 slack for the support hips, which sag
                              # ~0.055 rad under body weight before the policy compensates (audit).
# Head excursion caps around HOME (rad): neck_pitch, head_pitch, head_yaw, head_roll. The head
# may act as a counterweight, but bounded (v2 twisted it 116° and jammed head_roll).
HEAD_CAPS = (0.6, 0.6, 0.5, 0.25)
# Pushes: final ±0.15 m/s (sitstand's ±0.3 = 2× here). A single-leg stance has ~1 cm of CoM
# margin; v2's curves dropped exactly as the inherited ramp reached ±0.3 (composite 2.42 @1500
# → 1.70 @3000, falls 2.6 → 6.8), and nothing pushes a flamingo hard at deployment.
PUSH_SCALE = 0.5
# Trunk height targets (m above the env origin), MEASURED with the v2 policy in sim:
# two feet 116.3 mm, one foot 117.1 mm; the v3 target pose sits at 129 mm with a straighter
# support leg. z_one = 122 mm between them, std 20 mm: both score >= 0.88, the v3 kickstand
# crouch (94 mm) scores 0.14.
TRUNK_Z_TWO = 0.116
TRUNK_Z_ONE = 0.122
TRUNK_Z_STD = 0.02

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
    cfg.rewards["flamingo_height"] = RewardTermCfg(
        func=microduck_mdp.flamingo_trunk_height, weight=1.0,
        params={**common, "z_two": TRUNK_Z_TWO, "z_one": TRUNK_Z_ONE, "std": TRUNK_Z_STD},
    )
    cfg.rewards["flamingo_stillness"] = RewardTermCfg(
        func=microduck_mdp.flamingo_hold_stillness, weight=1.0,
        params={**common, "sensor_name": feet, "ang_vel_std": 0.5},
    )
    # Head: commandable but LIGHT — it is 38 % of the mass and a useful counterweight;
    # the head command keeps sitstand's stage-0 alive ranges only. Bounded by head_range.
    cfg.rewards["head_pose_tracking"].weight = 0.3

    # ── Costs ─────────────────────────────────────────────────────────────────
    cfg.rewards["non_sole_contact"] = RewardTermCfg(
        func=microduck_mdp.flamingo_non_sole_contact, weight=-2.0,
        params={"sensor_name": body_ground_cfg.name},
    )
    # Anti-jam stack (v2 lesson: five joints parked on their hard stops, commanded up to
    # 6.6 rad past them). All three are mjlab-style costs (>= 0) → NEGATIVE weights.
    #  - joint_limits: qpos within LIMIT_MARGIN of a HARD stop, every servo joint (the v3
    #    target keeps >= 0.08 rad from every stop; band 0.05 leaves slack for load sag).
    #  - action_over_limit: COMMANDED target past (stop + 0.3 rad) — the over-drive that
    #    turns a soft servo into a max-torque clamp. Same term/weight as the rollers env.
    #  - servo_stall: |torque| > 0.4 N·m at |vel| < 0.5 rad/s (velstand's term and weight).
    #    Holding the v3 pose needs <= 0.16 N·m statically, so a clean stance is free.
    cfg.rewards["joint_limits"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "margin": LIMIT_MARGIN},
    )
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty, weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty, weight=-0.05,
        params={"torque_thresh": 0.4, "vel_thresh": 0.5},
    )
    cfg.rewards["head_range"] = RewardTermCfg(
        func=microduck_mdp.flamingo_head_range_cost, weight=-5.0,
        params={"caps": HEAD_CAPS},
    )
    # Kept from sitstand (sim2real parity): action_rate_l2 (+ its -0.1 → -1.0 ramp),
    # joint_torque_rate_l2 (+ ramp), gentle_motion |a_z| (POSITIVE weight, self-negating),
    # body_ang_vel, angular_momentum, self_collisions, head_pose_tracking.

    # ── Pushes: halve the inherited sitstand ramp (see PUSH_SCALE) ───────────────
    if "push_magnitude" in cfg.curriculum:
        for st in cfg.curriculum["push_magnitude"].params["push_stages"]:
            st["velocity_range"] = {k: (lo * PUSH_SCALE, hi * PUSH_SCALE) for k, (lo, hi) in st["velocity_range"].items()}
    if "push_robot" in cfg.events:
        vr = cfg.events["push_robot"].params["velocity_range"]
        cfg.events["push_robot"].params["velocity_range"] = {k: (lo * PUSH_SCALE, hi * PUSH_SCALE) for k, (lo, hi) in vr.items()}

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
