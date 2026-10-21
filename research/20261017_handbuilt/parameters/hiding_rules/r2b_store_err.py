"""r2b diagnostic: is the previous-jump term of neurons 6 and 8 the store recursion's error after a record?
(1) How far the 2x2 store/detector recursion is from ARC's store and detector, overall and right after a record,
    and how much of the gap each source neuron explains (ARC's weight times ARC's activity).
(2) One-step fits (masked rule features, blind feedback, on-steps): store and detector from the 2x2 recursion vs
    ARC's actual store and detector vs recursions that add some of the store's other inputs; rms and jump coefficients."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, WH, lag, task_vars, sd_rule
import simlib
from r1b_masked import feats, MASK

np.set_printoptions(precision=2, suppress=True, linewidth=160)
X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
rec = lag(V["newmax"].astype(float)) > 0
for c, n in [(0, 2), (1, 4)]:
    err = Hp[..., n] - sd[..., c]
    print(f"neuron {n}: rule minus ARC rms {np.sqrt((err ** 2).mean()):.2f}; after a record {np.sqrt((err[rec] ** 2).mean()):.2f}; "
          f"mean after record {err[rec].mean():+.2f}")
    # the input ARC's neuron n gets from sources other than 2, 4 (one step back from the states it feeds)
    Hpp = np.concatenate([np.zeros((len(X), 1, 16)), Hp[:, :-1]], 1)
    contrib = {k: WH[n, k] * Hpp[..., k] for k in range(16) if k not in (2, 4)}
    tot = sum(contrib.values())
    print("   other inputs, mean |.| (all, after record): " + ", ".join(
        f"{k}:{np.abs(v).mean():.2f}/{np.abs(v[rec]).mean():.2f}" for k, v in sorted(contrib.items(), key=lambda z: -np.abs(z[1]).mean())[:8])
        + f"; total {np.abs(tot).mean():.2f}")


def onestep(store_det, label):
    Fn = feats(X).copy(); Fn[..., 1:3] = store_det
    blind = Hp[..., HID] - lag(X)[..., None] * WI[HID]
    out = []
    for jump in [True, False]:
        rs = []; js = []
        for i, n in enumerate(HID):
            m = MASK[i].copy()
            if not jump: m[3] = False
            y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
            A = np.concatenate([Fn[..., m], blind], -1).reshape(-1, m.sum() + 4)[on]
            c = np.linalg.lstsq(A, y[on], rcond=None)[0]
            rs.append(np.sqrt(((A @ c - y[on]) ** 2).mean())); js.append(c[3] if m[3] else np.nan)
        out.append(f"jump={'y' if jump else 'n'} rms {np.array(rs)} jump coef {np.array(js)}")
    print(f"{label:34s} " + " | ".join(out))


onestep(sd, "2x2 recursion")
onestep(Hp[..., [2, 4]], "ARC's store and detector")


def rec_with(extra):
    """store/detector recursion with ARC's weights from the listed extra sources, fed ARC's activity of those sources."""
    B = len(X); s = np.zeros(B); d = np.zeros(B); out = np.zeros((B, 10, 2))
    for t in range(10):
        hp = H[:, t - 1] if t else np.zeros((B, 16))
        es = sum(WH[2, k] * hp[:, k] for k in extra) if extra else 0
        ed = sum(WH[4, k] * hp[:, k] for k in extra) if extra else 0
        s, d = (np.maximum(WH[2, 2] * s + WH[2, 4] * d + WI[2] * X[:, t] + es, 0),
                np.maximum(WH[4, 2] * s + WH[4, 4] * d + WI[4] * X[:, t] + ed, 0))
        out[:, t] = np.stack([s, d], -1)
    return np.concatenate([np.zeros((B, 1, 2)), out[:, :-1]], 1)


print("ARC's weights into store (2) and detector (4) from each source:")
print("  ", WH[2]); print("  ", WH[4])
onestep(rec_with(HID), "+ hiding sources")
onestep(rec_with([k for k in range(16) if k not in (2, 4)]), "+ all sources (sanity: = ARC)")
onestep(rec_with([k for k in range(16) if k not in (2, 4) + tuple(HID)]), "+ non-hiding sources")


def rec_blind(extra):
    """as rec_with, but each extra source k is read blind to the previous number: h_k(t-1) - w_in,k x(t-1)."""
    B = len(X); s = np.zeros(B); d = np.zeros(B); out = np.zeros((B, 10, 2))
    for t in range(10):
        hp = H[:, t - 1] if t else np.zeros((B, 16)); xp = X[:, t - 1] if t else np.zeros(B)
        es = sum(WH[2, k] * (hp[:, k] - WI[k] * xp) for k in extra)
        ed = sum(WH[4, k] * (hp[:, k] - WI[k] * xp) for k in extra)
        s, d = (np.maximum(WH[2, 2] * s + WH[2, 4] * d + WI[2] * X[:, t] + es, 0),
                np.maximum(WH[4, 2] * s + WH[4, 4] * d + WI[4] * X[:, t] + ed, 0))
        out[:, t] = np.stack([s, d], -1)
    return np.concatenate([np.zeros((B, 1, 2)), out[:, :-1]], 1)


print("blind hiding sources into the store and detector:")
a = rec_blind(HID)
for lab, sdx in [("+ blind hiding", a), ("+ blind hiding, store only", np.stack([a[..., 0], sd[..., 1]], -1))]:
    for c, n in [(0, 2), (1, 4)]:
        e = Hp[..., n] - sdx[..., c]
        print(f"   {lab}: neuron {n} rms vs ARC {np.sqrt((e ** 2).mean()):.2f}, after record {np.sqrt((e[rec] ** 2).mean()):.2f}")
    onestep(sdx, lab)
