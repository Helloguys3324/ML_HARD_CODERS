"""Simulate the evaluator's RS decoder to confirm the error tolerance.

We reconstruct the scenario with n=5000, k=4252, p=65521 and verify how many
wrong labels the Gao decoder can tolerate before recovery fails.
"""
import sys, time
import numpy as np
sys.path.insert(0, 'challenges/task_romania/data')
from evaluation_script import _decode, _evaluate

p, n, k = 65521, 5000, 4252
rng = np.random.default_rng(42)

# random degree<k polynomial
coeffs = rng.integers(0, p, size=k).astype(np.int64)
x = np.arange(1, n + 1, dtype=np.int64)

t0 = time.time()
# Horner evaluation of f at all x
c_true = np.zeros(n, dtype=np.int64)
for d in range(k - 1, -1, -1):
    c_true = (c_true * x + coeffs[d]) % p
print(f"evaluated f ({time.time()-t0:.1f}s)")

inv = np.zeros(n + 2, dtype=np.int64)
for d in range(1, n + 2):
    inv[d] = pow(d, p - 2, p)

xs = np.arange(1, n + 1, dtype=np.int64)

def try_decode(err_rate):
    ys = c_true.copy()
    nerr = int(err_rate * n)
    idx = rng.choice(n, nerr, replace=False)
    ys[idx] = (ys[idx] + rng.integers(1, p, size=nerr)) % p  # wrong values
    f = _decode(xs, ys, k, p, inv)
    if f is None:
        return False, -1
    agree = int((_evaluate(f, xs, p) == ys).sum())
    # is f the TRUE polynomial? compare evaluation to true codeword
    fvals = _evaluate(f, xs, p)
    correct_f = int((fvals == c_true).sum())
    return correct_f == n, agree

for rate in [0.05, 0.07, 0.074, 0.075, 0.076, 0.08, 0.09]:
    ok, agree = try_decode(rate)
    print(f"err_rate={rate:.3f} ({int(rate*n)} errors): recovered_true={ok} agree={agree}")
print('done')