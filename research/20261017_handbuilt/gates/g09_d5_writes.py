"""Does neuron 5's history part D5 become exact once the memory's writes are named?
Hypothesis: the memory is close to a linear register driven by the non-memory neurons, and those neurons' values are
hinges of each past number against switch points ((r_j - x_s)^+ for hiding neurons, (x_s - r_4)^+ for the detector)
and the gates' own past values. Role features treat each past number as linear in its value, which misses the hinges.
Feature sets for D5(t), fitted on seed 6 (200,000 lists), scored on seed 7:
  roles            - rolefeat role features + x_t, x_{t+1}, own previous value (as g06/g08)
  roles+hinges     - plus, for each age 1..9, the five hinge values of that past number (true switch points)
  roles+hinges+gates - plus the two gates' values one step after each past number, by age
  roles+hinges+gates, by step - same features, coefficients fitted separately for each step t"""
import numpy as np
from gatelib import WI, WH, lists, run
from rolefeat import role_features
from gaterules import HJ, HINGE_SIGN


def build(X):
    L, P, H = run(X)
    keep = np.zeros((len(X), 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    # hinge values at each step s (number x_s against switch points from the state before it)
    HG = np.zeros((len(X), 10, 5))
    for s in range(10):
        u = (H[:, s - 1] if s > 0 else np.zeros((len(X), 16))) @ WH.T
        for k, j in enumerate(HJ):
            r = -u[:, j] / WI[j]
            HG[:, s, k] = np.maximum(X[:, s] - r, 0) if HINGE_SIGN[j] > 0 else np.maximum(r - X[:, s], 0)
    out = []
    for t in range(1, 9):
        F = role_features(X, t, keep)[0]
        base = np.c_[F, X[:, t], X[:, t + 1], H[:, t, 5]]
        hz = np.zeros((len(X), 9 * 5)); gz = np.zeros((len(X), 9 * 2))
        for a in range(1, 10):
            s = t - a
            if s >= 0:
                hz[:, (a - 1) * 5:(a) * 5] = HG[:, s]
                gz[:, (a - 1) * 2] = H[:, s + 1, 5]; gz[:, (a - 1) * 2 + 1] = H[:, s + 1, 9]
        D = P[:, t + 1, 5] - sum(WH[5, j] * H[:, t, j] for j in HJ)
        out.append((t, base, hz, gz, D))
    return out


A, B = build(lists(6)), build(lists(7))
sets = {"roles": lambda r: r[1], "roles+hinges": lambda r: np.c_[r[1], r[2]],
        "roles+hinges+gates": lambda r: np.c_[r[1], r[2], r[3]]}
for name, f in sets.items():
    Aa = np.concatenate([f(r) for r in A]); ba = np.concatenate([r[4] for r in A])
    co = np.linalg.lstsq(Aa, ba, rcond=None)[0]
    Ab = np.concatenate([f(r) for r in B]); bb = np.concatenate([r[4] for r in B])
    e = bb - Ab @ co
    tb = np.concatenate([np.full(len(r[4]), r[0]) for r in B])
    print(f"{name:22s} {Aa.shape[1]:4d} coefficients  rms error {np.sqrt((e ** 2).mean()):.3f}  by step: " +
          " ".join(f"{t}:{np.sqrt((e[tb == t] ** 2).mean()):.2f}" for t in range(1, 9)))
# per-step fit
f = sets["roles+hinges+gates"]; errs = []
for ra, rb in zip(A, B):
    co = np.linalg.lstsq(f(ra), ra[4], rcond=None)[0]; e = rb[4] - f(rb) @ co; errs.append(e)
print("roles+hinges+gates, separate coefficients per step: rms by step " + " ".join(f"{t}:{np.sqrt((e ** 2).mean()):.2f}" for t, e in zip(range(1, 9), errs)) +
      f"  overall {np.sqrt(np.mean(np.concatenate(errs) ** 2)):.3f}")
