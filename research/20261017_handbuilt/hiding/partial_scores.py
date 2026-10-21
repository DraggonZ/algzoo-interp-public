"""What one partial switch does to the scores. For lists where the number x_j at step j partly switches the four off
(some but not all of 1, 6, 7, 8 at zero), with j one step after a record (K = 1) or later (K >= 2) or at a record that
leaves some of the four on (K = 0): undo the clipping at that one step only (the off ones keep their negative
pre-activation), everything else unchanged. Split by the role of x_j at the end (the answer = second-largest, the
third, fourth or lower; at K = 0 x_j is a record, so: the final largest or overtaken later). Reported, medians:
the final-score change (undo minus normal) at x_j's position, at the largest's, at the answer's, and at the other
positions; x_j's vote per unit (score change at its position minus the mean change elsewhere, when x_j is nudged)
with and without the clip; and accuracy. 300,000 lists, seed 7."""
import numpy as np
from common import WI, WH, WO, C, lists, run

X = lists(7, 300_000); B = len(X); y = X.argsort(1)[:, -2]; o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
L0, P0, H0 = run(X)
pm = np.maximum.accumulate(np.c_[np.zeros(B), X], 1)[:, :-1]; rec = X > pm
lr = np.full(B, -1); Kst = np.zeros((B, 10), int)
for t in range(10):
    lr = np.where(rec[:, t], t, lr); Kst[:, t] = np.where(lr >= 0, t - lr, -1)
off = (P0[:, :, C] <= 0).sum(2); part = (off > 0) & (off < 4)


def run_undo(Xs, js, eps_pos=None, eps=0.0):
    Xs = Xs.copy()
    if eps_pos is not None: Xs[np.arange(len(Xs)), eps_pos] += eps
    def hook(t, pre, hn, hprev):
        m = js == t
        if m.any():
            hn = hn.copy()
            hn[np.ix_(m, C)] = pc   # all four take their pre-activation (the on ones already equal it)
        return hn
    return run(Xs, hook=hook)[0]


def run_plain(Xs, eps_pos=None, eps=0.0):
    Xs = Xs.copy()
    if eps_pos is not None: Xs[np.arange(len(Xs)), eps_pos] += eps
    return run(Xs)[0]


rng = np.random.default_rng(1)
for kcls in ("K0", "K1", "K2+"):
    km = {"K0": Kst == 0, "K1": Kst == 1, "K2+": Kst >= 2}[kcls] & part
    has = km.any(1); j = np.argmax(km, 1)        # first such step
    ii = np.where(has)[0]; ii = ii[rng.permutation(len(ii))[:60000]]
    jj = j[ii]; Xs = X[ii]
    roles = {}
    if kcls == "K0":
        roles["record stays the largest"] = lar[ii] == jj
        roles["record overtaken later"] = lar[ii] != jj
    else:
        roles["x_j is the answer"] = sec[ii] == jj; roles["x_j is the third"] = thi[ii] == jj
        roles["x_j is fourth or lower"] = (sec[ii] != jj) & (thi[ii] != jj) & (lar[ii] != jj)
    Ln = L0[ii]; Lu = run_undo(Xs, jj); e = 1e-3
    Ln_e = run_plain(Xs, jj, e); Lu_e = run_undo(Xs, jj, jj, e)
    a = np.arange(len(ii))
    def vote(L1, L2):
        d = (L2 - L1) / e; own = d[a, jj]; rest = (d.sum(1) - own) / 9; return own - rest
    vn, vu = vote(Ln, Ln_e), vote(Lu, Lu_e); d = Lu - Ln
    yy, ll = y[ii], lar[ii]
    print(f"== partial switch at {kcls}: {has.mean():.2f} of lists have one; {len(ii)} used")
    for lab, m in roles.items():
        oth = np.ones((len(ii), 10), bool); oth[a, jj] = False; oth[a, yy] = False; oth[a, ll] = False
        print(f"  {lab:28s} ({m.mean():.2f}): accuracy {np.mean(Ln[m].argmax(1) == yy[m]):.3f} -> {np.mean(Lu[m].argmax(1) == yy[m]):.3f}; "
              f"score change at x_j {np.median(d[m, jj[m]]):+6.2f}, largest {np.median(d[m, ll[m]]):+6.2f}, answer {np.median(d[m, yy[m]]):+6.2f}, "
              f"others {np.median(d[m][oth[m]]):+6.2f}; vote of x_j per unit {np.median(vn[m]):6.1f} -> {np.median(vu[m]):6.1f}", flush=True)
