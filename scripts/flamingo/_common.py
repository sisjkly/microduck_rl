"""Shared helpers for the flamingo evaluation tools (CPU, closed loop, commands forced).

The env is the TRAINING env of the task (same actuator model, DR, delays, backlash), with:
  - curricula removed (eval conditions are fixed by the arguments, not by a step counter),
  - command resampling disabled (the scripts write the command every step),
  - the one-foot spawn probability and the push magnitude set explicitly.
Which servo / mass model that is depends on the branch: `flamingo` = official XL330 build.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Flamingo-Flat-Backlash-MicroDuck"


def make_env(num_envs: int, spawn_prob: float, push: float, task: str = TASK,
             episode_s: float = 30.0) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.curriculum = {}
    cfg.events["flamingo_spawn"].params["prob"] = spawn_prob
    if push > 0:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
    else:
        cfg.events.pop("push_robot", None)
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    cfg.episode_length_s = episode_s
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def load_policy(env: ManagerBasedRlEnv, ckpt: str, task: str = TASK):
    """(wrapped env, inference policy) — the policy applies the checkpoint's obs normalizer."""
    agent_cfg = load_rl_cfg(task)
    alg = asdict(agent_cfg)
    alg["upload_model"] = False
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(wrapped, alg, None, "cpu")
    runner.load(ckpt, map_location="cpu")
    return wrapped, runner.get_inference_policy(device="cpu")


def set_flamingo_cmd(env: ManagerBasedRlEnv, flag, side) -> None:
    """twist = [flag, side, 0]: flag 0 two feet / 1 one foot; side +1 RIGHT foot down, -1 LEFT."""
    term = env.command_manager.get_term("twist")
    term.vel_command_b[:] = 0.0
    term.vel_command_b[:, 0] = flag
    term.vel_command_b[:, 1] = torch.as_tensor(flag, dtype=torch.float32) * side


def alternating_sides(n: int) -> torch.Tensor:
    return torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
