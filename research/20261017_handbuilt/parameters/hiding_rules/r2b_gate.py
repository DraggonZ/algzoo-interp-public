"""r2b diagnostic: can the late gate (neuron 9) be computed from the store and detector with ARC's own weights, and can
that computed gate, times ARC's weight from 9 into each hiding neuron, replace the record-aftermath features
(jump two back, new largest two back) at no constant cost?"""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, WH, lag, task_vars, sd_rule
import simlib
from r1b_masked import feats, MASK

np.set_printoptions(precision=3, suppress=True, linewidth=160)
X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
for g in [9, 5]:
    print(f"neuron {g}: input weight {WI[g]:+.2f}; ARC weights in: " + ", ".join(f"{k}:{WH[g, k]:+.2f}" for k in np.argsort(-np.abs(WH[g]))[:8]))
    print(f"   mean activity {H[..., g].mean():.2f}, on {np.mean(H[..., g] > 0):.2f}; mean |weight x activity| by source: " + ", ".join(
        f"{k}:{np.abs(WH[g, k] * Hp[..., k]).mean():.2f}" for k in np.argsort(-np.abs(WH[g] * Hp.mean((0, 1))))[:8]))
print("ARC weights from 5 and 9 into 1, 6, 7, 8:", WH[np.ix_(HID, [5, 9])].T)


def gate_rule(g, srcs, own=True):
    """gate g from the store and detector (rule recursion) and its own previous value, ARC's weights; activity (B,10)."""
    B = len(X); out = np.zeros((B, 10)); prev = np.zeros(B)
    s = np.concatenate([sd_rule(X)[:, 1:], np.zeros((B, 1, 2))], 1)    # s[:, t] = store/det at step t (not lagged)
    sdl = sd_rule(X)                                                    # sdl[:, t] = store/det at step t-1
    for t in range(10):
        pre = WI[g] * X[:, t] + (WH[g, g] * prev if own else 0)
        pre = pre + WH[g, 2] * sdl[:, t, 0] + WH[g, 4] * sdl[:, t, 1]
        prev = np.maximum(pre, 0); out[:, t] = prev
    return out


print("\nfirst attempt below (gate from store and detector) is wrong: the gate reads hiding neurons 7, 8, 1, see weights above")
print("second attempt: gate 9 from ARC's hiding activity (1, 6, 7, 8), its own previous value, and the input, ARC's weights")
gsrc = HID + [9]
G9 = np.maximum(WI[9] * X + sum(WH[9, k] * Hp[..., k] for k in gsrc), 0)       # (B, 10): gate at step t
print(f"   gate from hiding + own vs ARC: rms {np.sqrt(((G9 - H[..., 9]) ** 2).mean()):.2f}; corr {np.corrcoef(G9.ravel(), H[..., 9].ravel())[0, 1]:.3f}")
xpl = lag(X)
G9b = np.maximum(WI[9] * X + sum(WH[9, k] * (Hp[..., k] - (WI[k] * xpl if k in HID else 0)) for k in gsrc), 0)
print(f"   gate from blind hiding + own vs ARC: rms {np.sqrt(((G9b - H[..., 9]) ** 2).mean()):.2f}; corr {np.corrcoef(G9b.ravel(), H[..., 9].ravel())[0, 1]:.3f}")
G9a = np.maximum(WI[9] * X + sum(WH[9, k] * Hp[..., k] for k in range(16)), 0)
print(f"   gate from all sources (sanity) rms {np.sqrt(((G9a - H[..., 9]) ** 2).mean()):.4f}")
for k in [14, 11, 10, 3, 0, 12, 13, 15]:
    print(f"   memory {k} into gate: corr of its contribution with the hiding contribution "
          f"{np.corrcoef((WH[9, k] * Hp[..., k]).ravel(), sum(WH[9, j] * Hp[..., j] for j in HID).ravel())[0, 1]:+.2f}, mean {(WH[9, k] * Hp[..., k]).mean():+.2f}")
gl2 = lag(G9)
blind = Hp[..., HID] - lag(X)[..., None] * WI[HID]
Fn = feats(X)
for lab, gl_, drop in [("hiding-built gate, free coef, no jump2/nl2", gl2, [4, 5]), ("hiding-built gate, free coef, no nl2", gl2, [5]),
                       ("ARC's gate, free coef, no jump2/nl2", Hp[..., 9], [4, 5])]:
    rs = []
    for i, n in enumerate(HID):
        m = MASK[i].copy(); m[drop] = False; y = (P[:, :, n] - WI[n] * X)
        A = np.concatenate([Fn[..., m], blind, gl_[..., None]], -1); A = A.reshape(-1, A.shape[-1]); on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A[on], y.reshape(-1)[on], rcond=None)[0]
        rs.append(f"{np.sqrt(((A[on] @ c - y.reshape(-1)[on]) ** 2).mean()):.3f} (coef {c[-1]:+.2f}, ARC {WH[n, 9]:+.2f})")
    print(f"   {lab:45s}", "; ".join(rs))

for g in [9]:
    gr = gate_rule(g, [2, 4])
    print(f"gate {g} rule vs ARC: rms {np.sqrt(((gr - H[..., g]) ** 2).mean()):.2f} (ARC activity rms {np.sqrt((H[..., g] ** 2).mean()):.2f}); "
          f"corr {np.corrcoef(gr.ravel(), H[..., g].ravel())[0, 1]:.3f}")
    print("   one list: ARC", H[0, :, g].round(2), "\n   rule", gr[0].round(2), "\n   newmax", V["newmax"][0])
    gl = lag(gr)                                      # previous step's gate, as the hiding neurons read it
    ga = Hp[..., g]
    blind = Hp[..., HID] - lag(X)[..., None] * WI[HID]
    Fn = feats(X)
    for lab, mode in [("masked rule", "base"), ("gate rule fixed (ARC weight), no jump2/nl2", "fixed"),
                      ("gate rule free coef, no jump2/nl2", "free"), ("ARC's gate fixed, no jump2/nl2", "arcfixed")]:
        rs = []
        for i, n in enumerate(HID):
            m = MASK[i].copy(); y = (P[:, :, n] - WI[n] * X)
            if mode != "base": m[4] = m[5] = False
            cols = [Fn[..., m], blind]
            if mode == "fixed": y = y - WH[n, g] * gl
            if mode == "arcfixed": y = y - WH[n, g] * ga
            if mode == "free": cols.append(gl[..., None])
            A = np.concatenate(cols, -1); A = A.reshape(-1, A.shape[-1]); on = (P[:, :, n] > 0).reshape(-1)
            c = np.linalg.lstsq(A[on], y.reshape(-1)[on], rcond=None)[0]
            rs.append(np.sqrt(((A[on] @ c - y.reshape(-1)[on]) ** 2).mean()))
            if mode == "free": rs[-1] = (rs[-1], c[-1] / WH[n, g] if WH[n, g] else np.nan)
        print(f"   {lab:45s}", rs)
