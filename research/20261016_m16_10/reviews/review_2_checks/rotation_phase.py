"""Review 2 check: is the copies' two-spike carry loss a timing error of the free rotation?

(A) Rotation number of the free map h -> ReLU(W_hh h) / |.|: iterate from a fixed start for 300 steps,
    drop 50, project unit states on the attractor's top-2 principal plane, average the phase advance per step.
(B) Mean phase advance per step over the actual carries (all 90 pairs, a theta grid weighted by the
    two-spike density, steps after the last spike), in each model's own top-2 plane.
Compare |advance(copy) - advance(base)| with the exact carry loss from z2_stages.npy. Weights-only quantities;
the theta grid is deterministic.
"""
import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

NG = 120
tg = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
wg = g_angle(tg)
wg = wg / wg.sum()
PAIRS = [(a, b) for a in range(10) for b in range(10) if a != b]


def plane(X, w=None):
    w = np.ones(len(X)) if w is None else w
    C = (X * w[:, None]).T @ X / w.sum()
    ev, V = np.linalg.eigh(C)
    return V[:, -2:]


def advance(P, seq):
    """seq: [steps, N, d] unit states; mean unwrapped phase advance per step."""
    z = seq @ P
    ph = np.arctan2(z[..., 1], z[..., 0])
    d = np.angle(np.exp(1j * (ph[1:] - ph[:-1])))
    return d


def rot_number(w_hh):
    h = np.abs(np.random.default_rng(0).standard_normal(16))
    hs = []
    for t in range(300):
        h = np.maximum(w_hh @ h, 0)
        h /= np.linalg.norm(h) + 1e-300
        hs.append(h)
    H = np.array(hs[50:])
    P = plane(H)
    return np.abs(advance(P, H[:, None, :]).mean())


def carry_advance(params):
    w_hi, w_hh, w_oh = params
    seqs, ws = [], []
    allst = []
    per_pair = []
    for p1, p2 in PAIRS:
        T = max(p1, p2)
        if T >= 8:
            continue
        h = np.zeros((NG, 16))
        traj = []
        for t in range(10):
            x = np.where(t == p1, np.cos(tg), np.where(t == p2, np.sin(tg), 0.0))
            h = np.maximum(h @ w_hh.T + x[:, None] * w_hi, 0)
            if t >= T:
                traj.append(h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-300))
        traj = np.array(traj)
        seqs.append(traj)
        allst.append(traj.reshape(-1, 16))
    P = plane(np.concatenate(allst))
    tot, wt = 0.0, 0.0
    for traj in seqs:
        d = advance(P, traj)                 # [steps-1, NG]
        tot += np.abs((d * wg[None]).sum(1)).sum()
        wt += d.shape[0]
    return tot / wt


names, acc, models = load_bench(("mult",))
st = np.load("/data/output/research/algzoo/20261016_run2/reports/z2_stages.npy", allow_pickle=True).item()
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
base = example()
rb, cb = rot_number(base[1]), carry_advance(base)
print(f"base: rotation number {rb / 2 / np.pi:.4f} cycles/step (period {2 * np.pi / rb:.2f}); carry advance {cb / 2 / np.pi:.4f} cycles/step")
rows = []
for nm, m in zip(names, models):
    rows.append([rot_number(m[1]), carry_advance(m), st[nm][0]["right_T_wrong_end"], 1 - st[nm][0]["right_T_right_end"] - st[nm][0]["wrong_T_right_end"]])
rows = np.array(rows)
np.save("rotation_phase.npy", rows)
for j, lab in ((0, "rotation number"), (1, "carry advance")):
    dv = np.abs(rows[:, j] - (rb if j == 0 else cb))
    for k, tl in ((2, "carry loss"), (3, "1 - Z2")):
        r = np.corrcoef(dv, rows[:, k])[0, 1]
        rs = np.corrcoef(np.argsort(np.argsort(dv)), np.argsort(np.argsort(rows[:, k])))[0, 1]
        print(f"|Δ {lab}| vs {tl}: Pearson r {r:+.3f} (R² {r * r:.2f}), Spearman {rs:+.3f}")
    print(f"   |Δ {lab}| in cycles/step by σ: " + ", ".join(f"σ={s}: median {np.median(dv[sig == s]) / 2 / np.pi:.4f}" for s in (0.005, 0.01, 0.02)))
