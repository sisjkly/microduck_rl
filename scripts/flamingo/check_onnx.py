"""Check an exported ONNX against its checkpoint: same I/O contract, same numbers.

    uv run python scripts/flamingo/check_onnx.py <model_N.pt> flamingo.onnx

Recomputes actor(normalizer(obs)) by hand from the checkpoint's state dict (rsl_rl
EmpiricalNormalization: (x - mean) / (std + 1e-2), then the ELU MLP) on random observations
and compares it with onnxruntime. A deployable export prints max |diff| ~1e-6; a large diff
means the normalizer was not baked in (never convert by hand — use scripts/export.py).
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

OBS_LEN, ACTION_LEN, NORM_EPS = 61, 14, 1e-2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("onnx")
    ap.add_argument("--samples", type=int, default=256)
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx)
    ins, outs = sess.get_inputs(), sess.get_outputs()
    print("onnx inputs ", [(i.name, i.shape) for i in ins])
    print("onnx outputs", [(o.name, o.shape) for o in outs])
    assert ins[0].shape[-1] == OBS_LEN and outs[0].shape[-1] == ACTION_LEN, "expected 61 -> 14"

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    x = torch.randn(a.samples, OBS_LEN) * 0.5
    h = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + NORM_EPS)
    for j, i in enumerate(layers):
        h = h @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]
        if j < len(layers) - 1:
            h = torch.nn.functional.elu(h)
    y = np.concatenate([sess.run(None, {ins[0].name: xi[None].numpy().astype(np.float32)})[0] for xi in x])
    diff = float(np.abs(h.numpy() - y).max())
    print(f"max |torch - onnx| = {diff:.2e}  ->  {'OK' if diff < 1e-4 else 'MISMATCH'}")
    return 0 if diff < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
