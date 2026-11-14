#!/usr/bin/env python3
"""Build the final self-contained main.py for task_bolivia from a saved UAP.

Usage: python build_main.py <uap.npy> <out_main.py>
The UAP is expected to be a float32 (32000,) array (any norm; we renormalize).
"""
import base64
import sys

import numpy as np

uap_path = sys.argv[1] if len(sys.argv) > 1 else "uap_rain.npy"
out_path = sys.argv[2] if len(sys.argv) > 2 else "main.py"

d = np.load(uap_path).astype(np.float32).reshape(-1)
assert d.shape == (32000,), f"unexpected delta shape {d.shape}"
d = d.astype(np.float64)
d = d / np.linalg.norm(d)  # unit direction


def fmt(x):
    # compact scientific with enough precision but not excessive
    return f"{x:.9e}"


b64 = base64.b64encode(d.astype(np.float32).tobytes()).decode("ascii")

template = '''"""Reproduce the recurring 'rain' failure for The Forest Went Quiet (task_bolivia).

Self-contained: reads test-input.npz, adds a small universal perturbation that
shifts the deployed classifier toward 'rain', writes test-output.npz.

The edit energy is kept <= 1% of each recording and output samples stay in
(-1, 1). Deterministic (no randomness).
"""
import base64

import numpy as np

# Universal adversarial perturbation toward class 'rain' (unit-norm direction).
_UAP = np.frombuffer(base64.b64decode({B64!r}), dtype=np.float32).astype(np.float64)
_UAP = _UAP / np.linalg.norm(_UAP)  # ensure unit norm

_C = 0.0999  # sqrt(0.01) -> energy fraction C^2 = 0.998%
_TARGET_AMP = 0.9999


def transform(audio):
    x = audio.astype(np.float64)
    d = _UAP
    # per-recording scale for 1% energy: s = C * ||x||
    xn = np.sqrt((x * x).sum(-1, keepdims=True))
    s_energy = (_C * xn).reshape(-1)
    # amplitude headroom so output stays strictly within (-1, 1):
    # for d>0 need s < (T-x)/d ; for d<0 need s < (-T-x)/d
    T = _TARGET_AMP
    lim_pos = np.where(d > 0, (T - x) / d, np.inf)
    lim_neg = np.where(d < 0, (-T - x) / d, np.inf)
    lim = np.minimum(lim_pos.min(-1), lim_neg.min(-1))
    s = np.minimum(s_energy, lim * 0.999)
    out = x + s[:, None] * d
    return out.astype(np.float32)


def main():
    with np.load("test-input.npz", allow_pickle=False) as data:
        audio = data["audio"]
        sample_rate = data["sample_rate"]
        clip_ids = data["clip_ids"]
    np.savez("test-output.npz", audio=transform(audio),
             sample_rate=sample_rate, clip_ids=clip_ids)


if __name__ == "__main__":
    main()
'''

main_py = template.replace("{B64!r}", repr(b64))
with open(out_path, "w") as f:
    f.write(main_py)
print(f"wrote {out_path} ({len(main_py)} bytes) from {uap_path}")
print("delta unit norm:", np.linalg.norm(d))