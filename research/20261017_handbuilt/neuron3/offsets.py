"""Reading (b) check: share of new mistakes by offset (named position - true position), against the far
offsets (7-9 apart) where an eight-dimensional rotating code aliases (../bridge/README.md ideal angles)."""
from n3lib import *
from sub import quantities, fit_cells, run_mod
Xf, Xt = lists(21, 200_000), lists(101, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
S0, P0, H0 = run(Xt); yt = Xt.argsort(1)[:, -2]; ok0 = S0.argmax(1) == yt
def noisy(j, sd, seed=0):
    rng = np.random.default_rng(seed); h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; pre[:, j] += sd * rng.standard_normal(len(Xt)); h = np.maximum(pre, 0)
    return h @ WO.T
def linear3():
    h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; hn = np.maximum(pre, 0); hn[:, 3] = pre[:, 3]; h = hn
    return h @ WO.T
Hm = H0[:, 9].copy(); Hm[:, 3] = Pf[:, 9, 3].clip(0).mean()
cases = {"network's own mistakes": S0, "S3 hiding difference flattened": run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1), "none"),
         "neuron 3 readout at its mean": Hm @ WO.T, "neuron 3 linear": linear3(), "matched noise in 3": noisy(3, 3.04), "matched noise in 13": noisy(13, 2.18), "matched noise in 12": noisy(12, 1.79)}
print("share of (new) mistakes by |named - true| = 1, 2, 3-6, 7-9")
for nm, Sx in cases.items():
    a = Sx.argmax(1); new = ~ok0 if nm.startswith("network") else ok0 & (a != yt); d = np.abs(a[new] - yt[new])
    print(f"  {nm:32s} n={new.sum():6d}  1: {(d == 1).mean() * 100:4.0f}%  2: {(d == 2).mean() * 100:4.0f}%  3-6: {((d >= 3) & (d <= 6)).mean() * 100:4.0f}%  7-9: {(d >= 7).mean() * 100:4.0f}%")
