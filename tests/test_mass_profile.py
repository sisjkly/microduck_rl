"""Mass-profile patch (robot/mass_profile.py): totals, physics validity, scope.

Locks in that every robot variant gets the same 15 servo housings patched,
that the patched inertias stay physical, that "official" is an exact no-op,
and that microduck_constants' spec getters actually apply the active profile.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot import mass_profile as mp

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck"
VARIANTS = [
    "robot_walk.xml", "robot_groundcontact.xml", "robot_allcollisions.xml",
    "robot_groundcontact_rollers.xml", "robot_walk_backlash.xml",
    "robot_groundcontact_backlash.xml", "robot_allcollisions_backlash.xml",
    "robot_groundcontact_rollers_backlash.xml",
]


def _model(xml, profile):
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / xml))
    mp.apply_mass_profile(spec, profile)
    return spec.compile()


@pytest.mark.parametrize("xml", VARIANTS)
def test_every_variant_has_15_servo_housings(xml):
    m = _model(xml, None)
    assert sum(len(v) for v in mp.body_servo_geoms(m).values()) == 15


@pytest.mark.parametrize("xml", VARIANTS)
def test_ours_813g_total_and_physical_inertia(xml):
    m0, m = _model(xml, None), _model(xml, "ours_813g")
    assert m.body_mass.sum() - m0.body_mass.sum() == pytest.approx(0.075 + 0.0008, abs=1e-6)
    I = np.sort(m.body_inertia[1:], axis=1)
    assert np.all(I > 0) and np.all(I[:, 0] + I[:, 1] >= I[:, 2] * (1 - 1e-9))


def test_official_is_exact_noop():
    m0 = _model("robot_walk.xml", None)
    m1 = _model("robot_walk.xml", {"servo_extra_g": 0.0, "body_extra": {"trunk_base": {"g": 0.0}}})
    assert np.array_equal(m0.body_mass, m1.body_mass)
    assert np.allclose(m0.body_inertia, m1.body_inertia, atol=1e-15)
    assert np.allclose(m0.body_ipos, m1.body_ipos, atol=1e-15)


def test_head_extra_lands_on_head_only():
    m0, m = _model("robot_walk.xml", None), _model("robot_walk.xml", "ours_813g")
    head = m.body("neck").id
    assert (m.body_subtreemass[head] - m0.body_subtreemass[head]) * 1000 == pytest.approx(5 * 5.0 + 0.8, abs=1e-6)
    # Weighed head (~260 g) = everything from the head_pitch joint up.
    assert m.body_subtreemass[m.body("neck_pitch").id] * 1000 == pytest.approx(260.0, abs=3.0)
    # "com" mode keeps the head-roll body's CoM where the servo extras put it.
    b = m.body("jaw_soft").id
    assert np.linalg.norm(m.body_ipos[b] - m0.body_ipos[b]) < 2e-3


def test_patched_spec_still_resolves_renamed_geoms():
    """Regression: patching must not compile ``spec`` itself — that froze its
    name lookup, so the servo-housing geoms named afterwards by
    name_servo_collision_geoms were unfindable and mjlab's CollisionCfg editor
    crashed on spec.geom(name) (seen on the allcollisions model, 2026-10-21)."""
    pytest.importorskip("mjlab")
    from mjlab_microduck.robot.microduck_constants import name_servo_collision_geoms

    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "robot_allcollisions.xml"))
    mp.apply_mass_profile(spec, "ours_813g")
    name_servo_collision_geoms(spec)
    named = [g.name for g in spec.geoms if g.name.endswith("_servo_collision")]
    assert len(named) == 15
    assert all(spec.geom(n) is not None for n in named)


def test_constants_apply_active_profile(monkeypatch):
    pytest.importorskip("mjlab")
    from mjlab_microduck.robot import microduck_constants as mc

    monkeypatch.setenv(mp.PROFILE_ENV, "ours_813g")
    assert mc.get_walk_spec().compile().body_mass.sum() == pytest.approx(0.813, abs=1e-4)
    monkeypatch.setenv(mp.PROFILE_ENV, "official")
    assert mc.get_walk_spec().compile().body_mass.sum() == pytest.approx(0.7372, abs=1e-4)


@pytest.mark.parametrize("xml", VARIANTS)
def test_ours_835g_total_and_head(xml):
    m0, m = _model(xml, None), _model(xml, "ours_835g")
    assert m.body_mass.sum() * 1000 == pytest.approx(835.0, abs=1.0)
    if xml == "robot_walk.xml":
        # mouth fitted: head (head_pitch and up) weighed ~260 g without it + 22 g mouth
        assert m.body_subtreemass[m.body("neck_pitch").id] * 1000 == pytest.approx(282.0, abs=3.0)
    I = np.sort(m.body_inertia[1:], axis=1)
    assert np.all(I > 0) and np.all(I[:, 0] + I[:, 1] >= I[:, 2] * (1 - 1e-9))


def test_default_profile_is_the_robot_with_the_new_head():
    assert mp.DEFAULT_PROFILE == "ours_818g"


@pytest.mark.parametrize("xml", VARIANTS)
def test_ours_818g_total_and_head(xml):
    m = _model(xml, "ours_818g")
    assert m.body_mass.sum() * 1000 == pytest.approx(818.5, abs=1.0)
    if xml == "robot_walk.xml":
        assert m.body_subtreemass[m.body("neck_pitch").id] * 1000 == pytest.approx(265.5, abs=3.0)
