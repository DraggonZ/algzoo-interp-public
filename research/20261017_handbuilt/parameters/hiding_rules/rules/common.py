"""Shared pieces for rule modules scored by ../score_rule.py: list quantities, the account's features, and the
store/detector recursion. No side effects on import."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gao"))
import simlib  # noqa: E402
from simlib import task_vars  # noqa: E402,F401
from sim_groups import account_features, ANG  # noqa: E402,F401

WI, WH = simlib.WI, simlib.WH
HID = [1, 6, 7, 8]


def lag(a, k=1):
    out = np.zeros_like(a); out[:, k:] = a[:, :-k]; return out


def sd_rule(X):
    """Store (2) and detector (4) from their own 2x2 recursion with ARC's weights; previous step's values (B, 10, 2)."""
    B = len(X); s = np.zeros(B); d = np.zeros(B); out = np.zeros((B, 10, 2))
    for t in range(10):
        s, d = (np.maximum(WH[2, 2] * s + WH[2, 4] * d + WI[2] * X[:, t], 0),
                np.maximum(WH[4, 2] * s + WH[4, 4] * d + WI[4] * X[:, t], 0))
        out[:, t, 0], out[:, t, 1] = s, d
    return np.concatenate([np.zeros((B, 1, 2)), out[:, :-1]], 1)
