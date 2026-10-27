"""Per-build mass corrections applied to the Onshape-exported MJCF at load time.

The exported models describe the official build (15 Dynamixel XL330 housings,
737.2 g in sim). A clone with different servos / electronics is patched here
instead of hand-editing every ``robot_*.xml`` variant: every variant shares the
same body names and the same 15 servo housings, so one profile covers walk /
groundcontact / allcollisions / rollers and their ``_backlash`` twins.

Physics of a patch, per body (all in the body frame):
  * servo extra: a box of ``servo_extra_g`` at each XL330 housing mesh centroid
    (the compiled mesh geom pos IS the mesh centroid), oriented with the geom;
  * body extra: ``mode="com"`` grows the body with its CoM unchanged and its
    inertia scaled by the mass ratio (unknown distribution); ``mode="point"``
    adds a small sphere at ``pos`` (known location, e.g. a heavier PCB).
Mass, CoM (ipos) and the full inertia tensor are recombined with the parallel
axis theorem and written back as ``fullinertia``.

Import-light on purpose (mujoco + numpy only) so ``scripts/infer_policy.py``
can use it without pulling in mjlab / torch / warp.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

SERVO_MESH_NAME = "xl330"
# XL330 envelope (m): 20 x 34 x 26 mm. Used only for the extra mass's own
# inertia, which is ~1e-7 kg m^2 — orientation errors here are negligible.
_SERVO_BOX = (0.020, 0.034, 0.026)
_POINT_RADIUS = 0.01  # sphere radius for "point" extras (m)

# Name of the profile used by microduck_constants' spec getters. Env var wins
# so the CPU tools can A/B without editing code; None/"official" = no patch.
DEFAULT_PROFILE = "ours_818g"
PROFILE_ENV = "MICRODUCK_MASS_PROFILE"

PROFILES: dict[str, dict | None] = {
    "official": None,
    # Our clone (confirmed 2026-10-21): Feetech HD1910, 23 g each vs XL330 18 g
    # (kissqy's figure) -> +5 g x 15 housings (incl. the mouth) = +75 g, 812.2 g.
    # Weighed: whole robot 813 g; head ~260 g, which matches the head_pitch-and-up
    # subtree (neck_pitch body: 258.1 g with the servo extras) — so the head carries
    # no other extra mass. The 0.8 g remainder goes to the head-roll body.
    "ours_813g": {
        "servo_extra_g": 5.0,
        "body_extra": {"jaw_soft": {"g": 0.8, "mode": "com"}},
        "target_total_g": 813.0,
    },
    # Same robot with its mouth fitted (confirmed 2026-10-22): the 813 g weighing was
    # without the 22 g mouth, so real = 835 g and the head (head_pitch and up) ~282 g,
    # while the model head with its mouth parts is ~259 g. The extra ~22 g sits in the
    # head; where exactly is unknown, so it grows the head-roll body with CoM unchanged.
    # Walking with the mouth fitted matched sim attitude (pitch +0.6 deg vs +0.2..+1.8).
    "ours_835g": {
        "servo_extra_g": 5.0,
        "body_extra": {"jaw_soft": {"g": 22.8, "mode": "com"}},
        "target_total_g": 835.0,
    },
    # New head structure (confirmed 2026-10-27): whole robot 818.5 g, only the head changed
    # (-16.5 g vs ours_835g) -> the head extra drops from 22.8 g to 6.3 g, same place
    # (head-roll body, CoM unchanged); head (head_pitch and up) ~265.5 g.
    "ours_818g": {
        "servo_extra_g": 5.0,
        "body_extra": {"jaw_soft": {"g": 6.3, "mode": "com"}},
        "target_total_g": 818.5,
    },
    # kissqy/Microduck-kissqy "hd1910_820g_v1" (+5 g x 15; head +3 g; trunk takes
    # the rest to 820 g). Their deployed models were trained on this.
    "kissqy_820g": {
        "servo_extra_g": 5.0,
        "body_extra": {"jaw_soft": {"g": 3.0, "mode": "com"}, "trunk_base": {"g": 4.75682, "mode": "com"}},
        "target_total_g": 820.0,
    },
}


def active_profile_name() -> str | None:
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    return None if name in (None, "", "official", "none") else name


def _quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _box_inertia(mass: float, size) -> np.ndarray:
    a, b, c = size
    return mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def _shift(mass: float, r: np.ndarray) -> np.ndarray:
    """Parallel-axis term for a point mass at offset r."""
    return mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def body_servo_geoms(model: mujoco.MjModel) -> dict[int, list[int]]:
    """Visual XL330 housing geoms per body (contype=conaffinity=0 filters the
    collision copies so a housing is never counted twice)."""
    out: dict[int, list[int]] = {}
    for g in range(model.ngeom):
        mid = model.geom_dataid[g]
        if mid < 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.mesh(mid).name != SERVO_MESH_NAME:
            continue
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0:
            continue
        out.setdefault(int(model.geom_bodyid[g]), []).append(g)
    return out


def apply_mass_profile(spec: mujoco.MjSpec, profile: str | dict | None) -> mujoco.MjSpec:
    """Patch body inertials of ``spec`` in place according to ``profile``."""
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise KeyError(f"unknown mass profile {profile!r}; known: {sorted(PROFILES)}")
        profile = PROFILES[profile]
    if not profile:
        return spec

    # Compile a COPY: compiling ``spec`` itself freezes its name lookup, so geoms
    # renamed afterwards (name_servo_collision_geoms) become unfindable via
    # spec.geom(name) -> mjlab's CollisionCfg editor crashes.
    model = spec.copy().compile()
    servo_g = float(profile.get("servo_extra_g", 0.0)) / 1000.0
    body_extra = profile.get("body_extra", {})
    for name in body_extra:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise KeyError(f"mass profile body {name!r} not in model")

    servo_geoms = body_servo_geoms(model)
    bodies = {b.name: b for b in spec.bodies}
    for bid in range(1, model.nbody):
        name = model.body(bid).name
        geoms = servo_geoms.get(bid, []) if servo_g > 0 else []
        extra = body_extra.get(name)
        if not geoms and not extra:
            continue

        M = float(model.body_mass[bid])
        c = model.body_ipos[bid].astype(float).copy()
        R = _quat_to_mat(model.body_iquat[bid])
        I_c = R @ np.diag(model.body_inertia[bid]) @ R.T  # about CoM, body frame

        parts: list[tuple[float, np.ndarray, np.ndarray]] = []  # (m, pos, I_own)
        for g in geoms:
            Rg = _quat_to_mat(model.geom_quat[g])
            parts.append((servo_g, model.geom_pos[g].astype(float), Rg @ _box_inertia(servo_g, _SERVO_BOX) @ Rg.T))
        if extra:
            m_e = float(extra["g"]) / 1000.0
            if extra.get("mode", "com") == "com":
                # Same shape, denser: CoM fixed, inertia scales with mass.
                I_c = I_c * (M + m_e) / M
                M = M + m_e
            else:
                p = np.asarray(extra["pos"], dtype=float)
                parts.append((m_e, p, 0.4 * m_e * _POINT_RADIUS**2 * np.eye(3)))

        M_new = M + sum(m for m, _, _ in parts)
        c_new = (M * c + sum(m * p for m, p, _ in parts)) / M_new
        I_new = I_c + _shift(M, c - c_new)
        for m, p, I_own in parts:
            I_new = I_new + I_own + _shift(m, p - c_new)

        body = bodies[name]
        body.explicitinertial = True
        body.mass = M_new
        body.ipos = c_new
        body.fullinertia = [I_new[0, 0], I_new[1, 1], I_new[2, 2], I_new[0, 1], I_new[0, 2], I_new[1, 2]]

    target = profile.get("target_total_g")
    if target is not None:
        total = float(np.sum(spec.copy().compile().body_mass)) * 1000.0
        # rollers models are 0.4 g lighter than the rest (wheel parts) — allow 1 g.
        if abs(total - target) > 1.0:
            raise ValueError(f"mass profile total {total:.1f} g != target {target:.1f} g")
    return spec
