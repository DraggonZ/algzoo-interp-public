"""Does losing the hiding difference make the network name the number the second-largest displaced?
displaced = second-largest so far just before the step at which the final second-largest became second so far."""
from n3lib import *
from sub import quantities, fit_cells, run_mod

def displaced(X):
    n, T = X.shape; out = np.full(n, -1); tau = np.full(n, -1)
    order = X.argsort(1); sec = order[:, -2]
    m1i = np.full(n, -1); m2i = np.full(n, -1); prev2 = np.full((n, T), -1); cur2 = np.full((n, T), -1)
    for t in range(T):
        x = X[:, t]; v1 = np.where(m1i >= 0, X[np.arange(n), np.maximum(m1i, 0)], -np.inf)
        v2 = np.where(m2i >= 0, X[np.arange(n), np.maximum(m2i, 0)], -np.inf)
        prev2[:, t] = m2i
        new1 = x > v1; new2 = ~new1 & (x > v2)
        m2i = np.where(new1, m1i, np.where(new2, t, m2i)); m1i = np.where(new1, t, m1i); cur2[:, t] = m2i
    for i in range(n):
        ts = np.where(cur2[i] == sec[i])[0]; t0 = ts[0]; tau[i] = t0; out[i] = prev2[i, t0]
    return out, tau

Xf, Xt = lists(21, 200_000), lists(62, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
S0 = run(Xt)[0]; yt = Xt.argsort(1)[:, -2]; ok0 = S0.argmax(1) == yt
dsp, tau = displaced(Xt)
print(f"lists with a displaced number: {(dsp >= 0).mean() * 100:.0f}%; displaced is one position before the second: {(dsp == yt - 1).mean() * 100:.0f}%")
def noisy3(sd, seed=0):
    rng = np.random.default_rng(seed); h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; pre[:, 3] += sd * rng.standard_normal(len(Xt)); h = np.maximum(pre, 0)
    return h @ WO.T
cases = {"network's own mistakes": S0, "S3 hiding difference <- per-cell mean": run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1), "none"),
         "matched noise in neuron 3 (sd 3.04)": noisy3(3.04), "S1 gap formula": run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, "gap", range(1, 10), 1), "gap")}
for nm, Sx in cases.items():
    a = Sx.argmax(1); new = (~ok0) if nm.startswith("network") else (ok0 & (a != yt))
    d = a[new] == dsp[new]; oe = a[new] == yt[new] - 1
    print(f"{nm:40s} mistakes {new.sum():6d}: name the displaced number {d.mean() * 100:4.0f}%;  one earlier {oe.mean() * 100:4.0f}%, of which displaced {(d & oe).sum() / max(oe.sum(), 1) * 100:4.0f}%")
    # by kind of second
    for kn, sel in (("second came after the largest", yt > Xt.argsort(1)[:, -1]), ("second before the largest", yt < Xt.argsort(1)[:, -1])):
        mm = new & sel
        print(f"      {kn:32s} {mm.sum():6d} mistakes, displaced named {(a[mm] == dsp[mm]).mean() * 100:4.0f}%")
