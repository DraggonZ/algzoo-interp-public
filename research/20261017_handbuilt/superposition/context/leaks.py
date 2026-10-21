"""Do the smaller numbers act as linear leaks with a structured coefficient table? On each near tie the lead of the
second over the third is exactly linear on its piece: lead = sum_j c_j x_j. For the smaller numbers: (1) do first-order
leaks predict the flips when one smaller number is redrawn? (2) How much of c_j's variation is explained by positions
alone (cell means over (second's position, third's position, j)), and how much more by the largest's position?"""
import numpy as np
from flips import *
Xs = X[:200000]; Ss, Ts, Ls = S[:200000], T[:200000], L[:200000]; nn = torch.arange(len(Xs), device=DEV)
Jl = jac(ARC, Xs); c = (Jl[nn, Ss] - Jl[nn, Ts])                       # n x 10 lead coefficients
lead = (c * Xs).sum(1)
sm = small[:200000]
# (1) first-order prediction of flips from redrawing everything 'between the contenders'
Y = redraw(small)[:200000]; m = sm
pred_lead = lead + (c * (Y - Xs) * m).sum(1); d = run(ARC, Y).argmax(1); d00 = d0[:200000]; kk = keep[:200000]
real = (((d == Ss) != (d00 == Ss)) & kk).double().mean().item(); pr = (((pred_lead > 0) != (lead > 0)) & kk).double().mean().item()
print(f"redraw all smaller: flips real {100*real:.1f}%, first-order from the leaks {100*pr:.1f}%")
# (2) structure of c_j
cs = c[sm].cpu().numpy(); jj = torch.arange(10, device=DEV)[None].expand_as(c)[sm].cpu().numpy()
ps = Ss[:, None].expand_as(c)[sm].cpu().numpy(); pt = Ts[:, None].expand_as(c)[sm].cpu().numpy(); pl = Ls[:, None].expand_as(c)[sm].cpu().numpy()
xv = Xs[sm].cpu().numpy()
def r2(keys):
    k = np.zeros(len(cs), dtype=np.int64)
    for a in keys: k = k * 10 + a
    u, inv = np.unique(k, return_inverse=True); mean = np.bincount(inv, cs) / np.bincount(inv)
    return 1 - np.var(cs - mean[inv]) / np.var(cs), len(u)
print(f"smaller-number lead coefficients: median |c| {np.median(np.abs(cs)):.2f}; leak sum_j c_j x_j median |.| {torch.median(((c * Xs) * sm).sum(1).abs()).item():.2f} vs lead median |.| {lead.abs().median().item():.2f}")
for lab, keys in (("j only", [jj]), ("second, third, j", [ps, pt, jj]), ("second, third, largest, j", [ps, pt, pl, jj])):
    r, k = r2(keys); print(f"   share of c_j's variance explained by cell means over {lab:26s}: {r:.2f} ({k} cells)")
# by sign/size of the smaller number itself
for lab, mm in (("smaller number negative", xv < 0), ("positive", xv > 0)):
    print(f"   {lab}: median |c| {np.median(np.abs(cs[mm])):.2f}")
q = np.quantile(np.abs(cs), [0.5, 0.9, 0.99, 0.999]); print("quantiles of |c| 50/90/99/99.9%:", np.round(q, 2), "max", np.abs(cs).max().round(1))
cs_full = cs.copy()
for clip in (2.0, 5.0):
    cs = np.clip(cs_full, -clip, clip)
    out = [f"{lab}: {r2(keys)[0]:.2f}" for lab, keys in (("j", [jj]), ("second,third,j", [ps, pt, jj]), ("+largest", [ps, pt, pl, jj]), ("+sign of x_j", [ps, pt, pl, jj, (xv > 0).astype(int)]))]
    print(f"   clipped at ±{clip}: variance explained by " + " | ".join(out))
