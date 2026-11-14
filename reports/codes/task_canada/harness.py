"""Harness for task_canada: train linear (W,b) models and test against the vault."""
import numpy as np
from featurize import featurize
from open_vault import CLASSES, load_sealed, try_vault

DIM = 4096
NC = len(CLASSES)

def load_all():
    z = np.load('_cache.npz')
    return z['X'].astype(np.float64), z['y'].astype(np.int64), z['Xs'].astype(np.float64), z['lang']

def to_model(W, b):
    """W (NC,DIM), b (NC,) -> vault-format, padded with far-negative bias for safety."""
    return W, b

def fold_transform(W_, b_, A, mu):
    """Given model (W_, b_) trained on x' = A @ (x - mu), return W,b for raw x.
    y = W_ @ (A(x-mu)) + b_ = (W_@A) x + (b_ - W_@A@mu)."""
    W = W_ @ A
    b = b_ - W @ mu
    return W, b

def evaluate(W, b, Xs=None, verbose=True):
    if Xs is None:
        _, _, Xs, _ = load_all()
    scores = Xs @ W.T + b
    t = __import__('time').time()
    words = try_vault(scores, verbose=verbose)
    return words, scores

def save_model(W, b, path='model.npz'):
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32))

def class_bias_report(scores):
    pred = scores.argmax(1)
    return np.bincount(pred, minlength=NC)