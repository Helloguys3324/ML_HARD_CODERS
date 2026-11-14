"""Erasure + error Reed-Solomon decoding for task_romania flag recovery.

We know: sealed records are indexed 1..n (n=5000). The TRUE label y_i satisfies
  table[i, y_i] = f(i)   (mod p),  deg(f) < k  (k=4252, p=65521).

Equivalently, c = (c_1..c_n) with c_i = table[i, y_i] is a Reed-Solomon codeword:
  sum_{t=0}^{k} (-1)^t C(k, t) c_{j+t} == 0  (mod p)  for j = 1..(n-k).

We use parity-check linear algebra to solve for erased positions and, more
generally, to recover f when we have a mostly-correct hard label set.
"""
import hashlib
import numpy as np

# ---- reuse the evaluator's word-derivation logic ---------------------------
def word_from_coeffs(f, k, z):
    coeffs = np.zeros(k, dtype='<u2')
    coeffs[:len(f)] = f
    key = hashlib.sha256(coeffs.tobytes()).digest()
    if hashlib.sha256(key + b"door").digest()[:8] != bytes(z["tag"]):
        return None
    stream = hashlib.sha256(key + b"word").digest()
    return bytes(a ^ b for a, b in zip(bytes(z["sealed_word"]), stream)).decode("ascii")


def binom_row(k, p):
    """Return row [C(k,0), C(k,1), ..., C(k,k)] mod p via Pascal."""
    row = np.zeros(k + 1, dtype=np.int64)
    row[0] = 1
    for t in range(1, k + 1):
        row[t] = row[t - 1] * (k - t + 1) * pow(t, p - 2, p) % p
    return row


def build_H(n, k, p):
    """Parity-check matrix H (n-k) x n over F_p for RS eval at x=1..n, deg<k."""
    C = binom_row(k, p)  # length k+1
    m = n - k
    H = np.zeros((m, n), dtype=np.int64)
    # check j (0-indexed) uses positions x = j+1 .. j+1+k  => indices j .. j+k
    for j in range(m):
        for t in range(k + 1):
            i = j + t
            val = C[t] if t % 2 == 0 else (p - C[t]) % p  # (-1)^t
            H[j, i] = val
    return H


def debug_check_H(H, n, k, p, rng):
    """Verify H against a random codeword (deg < k)."""
    coeffs = rng.integers(0, p, size=k)
    x = np.arange(1, n + 1, dtype=np.int64)
    c2 = np.zeros(n, dtype=np.int64)
    for dd in range(k - 1, -1, -1):
        c2 = (c2 * x + coeffs[dd]) % p
    return (H @ c2) % p  # should be all zero


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n, k, p = 5000, 4252, 65521
    H = build_H(n, k, p)
    s = debug_check_H(H, n, k, p, rng)
    print("max |H c| mod p (should be 0):", int(np.abs(s).max()))