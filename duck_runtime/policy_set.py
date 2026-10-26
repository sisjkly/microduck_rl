"""The official policy set (walk + sit/stand + ground pick + one-shots) from its manifest.

Same contract as the official daemon (pollen-robotics/microduck 942e58e):
  - docs/policy-manifest.md (schema 2): `kind`, `command.encoding`, `duration_s`, `action_scale`,
    `ramp_s` / `unwind_s`, `mode`.
  - robotd/src/control.rs:421-760: one move at a time, none from the seat; a skill drives its
    constant twist with head/body zeroed; ground pick drives [cos 2*pi*phi, sin 2*pi*phi, 0] for
    period_s * end_phase; sit drives twist [1,0,0] (head live), rise [0,0,0] for rise_s; the
    previous action and the low-pass are shared across every network.
  - action scale: skill -> its own `action_scale`, else the gait's (velstand has no standing
    network, so the standing scale never applies); ground pick 1.0; sit/rise 1.0.

Our manifest extension (absent from the official manifest, so the official set runs as before):
  - per entry ``"lowpass": {"head": a, "legs": a}`` -> that network's own target low-pass
    (1.0 = straight through) instead of the global 0.5 / 0.7;
  - ``action_scale`` / ``lowpass`` on the perpetual walk entry -> the gait's scale and filter
    (unless --policy replaces the walk or --action-scale overrides it).
``duck_runtime.fetch_set kissqy`` writes such a manifest for kissqy's HD1910-trained set.

Not mirrored: per-skill servo gain ratios (P is fixed at 5 through NPD1, which only writes with
torque off) and roulade chaining (one key press = one roll). Roller-mode entries are skipped:
they need the roller accessory.

Directory layout = the Hub repo pollen-robotics/microduck-policies downloaded as is
(manifest.json next to the .onnx files).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .controller import Policy

GROUND_PICK_PERIOD_S = 4.0      # control.rs SkillTuning defaults
GROUND_PICK_END_PHASE = 0.7
GROUND_PICK_ACTION_SCALE = 1.0
SITSTAND_RAMP_S = 2.0
SITSTAND_RISE_S = 1.0
SITSTAND_ACTION_SCALE = 1.0
STANDING_ACTION_SCALE = 1.0     # control.rs Tuning default; what the kicks were tuned at


@dataclass
class Skill:
    name: str
    policy: Policy
    duration_s: float
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_scale: float | None = None          # None: the gait's
    encoding: str = "constant"                 # "constant" | "phase"
    period_s: float = GROUND_PICK_PERIOD_S
    unwind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    unwind_s: float = 0.0
    lowpass: tuple[float, float] | None = None  # (head, legs); None: the global filter


@dataclass
class SitStand:
    policy: Policy
    ramp_s: float = SITSTAND_RAMP_S
    rise_s: float = SITSTAND_RISE_S
    action_scale: float = SITSTAND_ACTION_SCALE
    lowpass: tuple[float, float] | None = None


@dataclass
class PolicySet:
    walk: Policy
    walk_file: str = ""
    sitstand: SitStand | None = None
    skills: dict[str, Skill] = field(default_factory=dict)   # priority / listing order
    walk_action_scale: float | None = None      # from the manifest's walk entry, if it says
    walk_lowpass: tuple[float, float] | None = None
    description: str = ""


def _lowpass(e: dict) -> tuple[float, float] | None:
    lp = e.get("lowpass")
    if not isinstance(lp, dict):
        return None
    head, legs = float(lp.get("head", 1.0)), float(lp.get("legs", 1.0))
    if not (0.0 < head <= 1.0 and 0.0 < legs <= 1.0):
        raise ValueError(f"{e.get('file')}: lowpass alphas must be in (0, 1], got {lp}")
    return head, legs


def _twist(v, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(float(x) for x in v)
    return default


def load_set(directory: str, walk: str | None = None, sitstand: str | None = None,
             loader=Policy) -> PolicySet:
    """Read ``directory/manifest.json``; ``walk`` / ``sitstand`` override those slots with a file."""
    with open(os.path.join(directory, "manifest.json")) as f:
        man = json.load(f)
    if man.get("schema_version") != 2 or man.get("obs_len") != 61 or man.get("action_len") != 14:
        raise ValueError(f"{directory}/manifest.json: need schema 2, 61 -> 14")
    path = lambda name: name if os.path.isabs(name) or os.path.exists(name) else os.path.join(directory, name)

    walk_file = walk
    walk_scale = walk_lp = None
    ss: SitStand | None = None
    skills: dict[str, Skill] = {}
    for e in man.get("policies", []):
        if e.get("mode", "walk") != "walk":
            continue                                    # roller accessory
        fname = e["file"]
        name = e.get("name", os.path.splitext(fname)[0])
        cmd = e.get("command") or {}
        enc = cmd.get("encoding", "constant")
        kind = e.get("kind")
        if not os.path.exists(path(fname)):
            continue                                    # a partial download is fine
        if kind == "perpetual":
            if e.get("slot") == "walk" and walk_file is None:
                walk_file = fname
                walk_scale = None if e.get("action_scale") is None else float(e["action_scale"])
                walk_lp = _lowpass(e)
        elif enc == "posture_flag":
            if sitstand is None:
                ss = SitStand(loader(path(fname)), ramp_s=float(e.get("ramp_s", SITSTAND_RAMP_S)),
                              rise_s=float(e.get("unwind_s", SITSTAND_RISE_S)),
                              action_scale=float(e.get("action_scale", SITSTAND_ACTION_SCALE)),
                              lowpass=_lowpass(e))
        elif enc == "phase":
            period = float(cmd.get("period_s", GROUND_PICK_PERIOD_S))
            end = float(cmd.get("end_phase", GROUND_PICK_END_PHASE))
            skills[name] = Skill(name, loader(path(fname)), duration_s=period * end, encoding="phase",
                                 period_s=period,
                                 action_scale=float(e.get("action_scale", GROUND_PICK_ACTION_SCALE)),
                                 lowpass=_lowpass(e))
        elif kind in ("episodic", "scripted") and enc == "constant":
            if e.get("duration_s") is None:
                continue
            twist = _twist(cmd.get("twist"))
            scale = e.get("action_scale")
            if scale is None and twist == (0.0, 0.0, 0.0):
                scale = STANDING_ACTION_SCALE
            skills[name] = Skill(name, loader(path(fname)), duration_s=float(e["duration_s"]),
                                 twist=twist,
                                 action_scale=None if scale is None else float(scale),
                                 unwind=_twist(cmd.get("idle")), unwind_s=float(e.get("unwind_s", 0.0)),
                                 lowpass=_lowpass(e))
    if walk_file is None:
        raise ValueError(f"{directory}: no walk policy (manifest slot 'walk' or --policy)")
    if sitstand is not None:
        ss = SitStand(loader(path(sitstand)))
    return PolicySet(loader(path(walk_file)), walk_file, ss, skills,
                     walk_action_scale=walk_scale, walk_lowpass=walk_lp,
                     description=str(man.get("description", "")))


def apply_walk_tuning(pset: PolicySet, tuning, action_scale_override: float | None = None):
    """The set's own gait settings onto ``tuning`` (kissqy: scale 1.0, low-pass 0.5 / 0.7);
    an explicit --action-scale wins. The official manifest states neither: tuning unchanged."""
    if pset.walk_action_scale is not None:
        tuning.action_scale = pset.walk_action_scale
    if pset.walk_lowpass is not None:
        tuning.head_lowpass, tuning.legs_lowpass = pset.walk_lowpass
    if action_scale_override is not None:
        tuning.action_scale = action_scale_override
    return tuning


# Keys for the terminal driver; anything else gets 4..9 in manifest order.
DEFAULT_KEYS = {"ground_pick": "g", "kick_left": "k", "kick_right": "l", "roulade": "r"}


def bind_keys(skills: dict[str, Skill]) -> dict[str, str]:
    keys, spare = {}, iter("456789")
    for name in skills:
        k = DEFAULT_KEYS.get(name) or next(spare, None)
        if k:
            keys[k] = name
    return keys
