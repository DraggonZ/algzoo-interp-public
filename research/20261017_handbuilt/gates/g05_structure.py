"""The gates read the arriving number x_t only through a few hinges.
Write u_j(t) = W_j . h(t-1), the drive neuron j gets from the state before x_t, so h_j(t) = ReLU(u_j + w_j x_t).
For neuron j with a large input weight (detector 4, hiding neurons 1, 6, 7, 8) define its switch point (reference)
r_j = -u_j / w_j: hiding neurons are on while x_t < r_j, the detector while x_t > r_4.
Then, exactly,
   pre_g(t+1) = D_g(t) + sum_{j in 1,4,6,7,8} W_gj |w_j| (x_t - r_j)^+ or (r_j - x_t)^+ ,
where D_g(t) collects every other neuron's push (memory, store, the gates themselves) plus w_g x_{t+1}.
Question: does D_g depend on x_t? If not (or only by a tiny slope), the gate's reading of x_t is fully described by the
hinges, and what is left to explain is D_g and the switch points, which depend only on the list before x_t.
Also: how much random error in each gate costs (noise added to its pre-activation at every step)."""
import numpy as np
from gatelib import WI, WH, lists, run, answer

X = lists(5); L, P, H = run(X); y = answer(X)
HJ = [1, 4, 6, 7, 8]
for g in (5, 9):
    print(f"\n===== neuron {g}")
    print("  hinge slopes W_gj |w_j|: " + ", ".join(f"{j}: {WH[g, j] * abs(WI[j]):+.2f}" for j in HJ))
    Ds, xs = [], []
    for t in range(1, 9):
        D = P[:, t + 1, g] - sum(WH[g, j] * H[:, t, j] for j in HJ)
        Ds.append(D); xs.append(X[:, t])
    # slope of D in x_t: compare D with D computed from the same state but x_t nudged (exact partial derivative)
    eps = 1e-4; dd = []
    for t in range(1, 9):
        hprev = H[:, t - 1]; u = hprev @ WH.T
        def Dof(xv):
            h_t = np.maximum(u + xv[:, None] * WI, 0)
            pre = h_t @ WH[g] + X[:, t + 1] * WI[g]
            return pre - sum(WH[g, j] * h_t[:, j] for j in HJ)
        dd.append((Dof(X[:, t] + eps) - Dof(X[:, t])) / eps)
    dd = np.concatenate(dd)
    print(f"  dD/dx_t: mean {dd.mean():+.3f}, sd {dd.std():.3f}, 1st-99th percentile {np.percentile(dd, 1):+.3f} .. {np.percentile(dd, 99):+.3f}")
    D = np.concatenate(Ds)
    print(f"  D_g: mean {D.mean():.2f}, sd {D.std():.2f}")
# noise tolerance: add N(0, s) to the gate's pre-activation at every step
print("\naccuracy with random noise on one gate's pre-activation (sd s, every step), 200,000 lists")
base = (L.argmax(1) == y).mean()
print(f"  intact {100 * base:.2f}%")
rng = np.random.default_rng(1)
for g in (5, 9):
    for s in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        noise = rng.standard_normal((len(X), 10)) * s
        def hook(t, hp, pre, h, g=g, noise=noise):
            h[:, g] = np.maximum(pre[:, g] + noise[:, t], 0); return h
        Ln = run(X, hook=hook)[0]
        print(f"  neuron {g}, s = {s:4.2f}: {100 * (Ln.argmax(1) == y).mean():.2f}%")
