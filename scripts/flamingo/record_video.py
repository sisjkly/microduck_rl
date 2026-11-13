"""Roll out a checkpoint through a scripted command cycle and render an MP4 (software GL).

    uv run python scripts/flamingo/record_video.py <ckpt> flamingo.mp4 [--push 0.15] [--num-envs 3]

Cycle: two feet 2 s → one foot RIGHT 5 s → two feet 3 s → one foot LEFT 5 s → two feet 3 s.
Envs are tiled side by side (each with its own tracking camera) under a caption strip.
Headless servers: needs libosmesa6 (apt install libosmesa6); MUJOCO_GL=osmesa is set here.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from _common import TASK, load_policy, make_env, set_flamingo_cmd  # noqa: E402

CYCLE = [(2.0, 0, 0), (5.0, 1, +1), (3.0, 0, 0), (5.0, 1, -1), (3.0, 0, 0)]
LABEL = {(0, 0): "TWO FEET", (1, 1): "ONE FOOT: RIGHT down", (1, -1): "ONE FOOT: LEFT down"}


def cmd_at(t):
    acc = 0.0
    for dur, flag, side in CYCLE:
        acc += dur
        if t < acc:
            return flag, side
    return 0, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--num-envs", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()

    torch.manual_seed(0)
    env = make_env(a.num_envs, 0.0, a.push, a.task)
    wrapped, policy = load_policy(env, a.ckpt, a.task)
    obs = wrapped.get_observations()
    mjm = env.sim.mj_model
    mjd = mujoco.MjData(mjm)
    renderer = mujoco.Renderer(mjm, a.size, a.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 0.55, -12.0, 120.0
    dt = env.step_dt
    every = max(1, round(1 / (a.fps * dt)))
    qpos = env.sim.data.qpos
    frames = []
    for k in range(int(sum(c[0] for c in CYCLE) / dt)):
        t = k * dt
        flag, side = cmd_at(t)
        set_flamingo_cmd(env, float(flag), torch.full((a.num_envs,), float(side)))
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wrapped.step(act)
        if k % every:
            continue
        tiles = []
        for i in range(a.num_envs):
            mjd.qpos[:] = qpos[i].cpu().numpy()
            mjd.qpos[0:2] -= env.scene.env_origins[i, 0:2].cpu().numpy()
            mujoco.mj_forward(mjm, mjd)
            cam.lookat[:] = [mjd.qpos[0], mjd.qpos[1], 0.08]
            renderer.update_scene(mjd, camera=cam)
            tiles.append(renderer.render())
        im = Image.fromarray(np.concatenate(tiles, axis=1))
        canvas = Image.new("RGB", (im.width, im.height + 34), (20, 20, 20))
        canvas.paste(im, (0, 34))
        ImageDraw.Draw(canvas).text(
            (10, 10), f"t={t:4.1f}s  cmd: {LABEL[(flag, side)]}   ({os.path.basename(a.ckpt)})", fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    imageio.mimsave(a.out, frames, fps=a.fps, codec="libx264", quality=7, macro_block_size=2)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
