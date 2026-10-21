"""Which earlier numbers does each gate read? The network has no biases, so every pre-activation is a piecewise-linear,
degree-1 homogeneous function of the list: pre(t) = sum_s g_s x_s exactly, with g_s = d pre / d x_s along the list's own
on/off pattern (Euler). We compute g_s exactly (forward-mode Jacobian) and group the earlier numbers by role.

Roles for the gate one step after x_t (so the gate's step is t+1), earlier numbers s < t:
  m1 = the largest before x_t, m2 = the second-largest before x_t, p = the value m1 overtook (if m1 overtook a positive
  record, p is that record; it is usually m2), other records (older beaten records that are not m2),
  kept (positive, not a record, passed neuron 5 one step after it, not m2), forgotten (positive, failed neuron 5),
  negative. Plus x_t itself and the next input x_{t+1}.
Check: sum_s g_s x_s equals the pre-activation (printed)."""
import numpy as np
from gatelib import WI, WH, lists, run, task_vars

X = lists(4, 100_000); B = len(X)
L, P, H = run(X); V = task_vars(X)
# forward-mode Jacobian J[t] = d h(t) / d x_s, shape (B, 10 inputs, 16)
J = np.zeros((B, 10, 16)); JS = []
for t in range(10):
    Jn = J @ WH.T
    Jn[:, t, :] += WI
    Jn *= (P[:, t] > 0)[:, None, :]
    J = Jn; JS.append(J.copy())
JS = np.stack(JS, 1)   # (B, step, input, neuron)

keep = np.zeros((B, 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
order_roles = ["x_t", "next x", "m1", "m2", "other record", "kept", "forgotten", "negative"]
for g in (5, 9):
    print(f"\n===== neuron {g}, one step after x_t: d pre / d x_s by role (mean, sd, share of |pre| explained)")
    for case in ("record", "not record, gate 5 on", "not record, gate 5 off, x>0"):
        acc = {r: [] for r in order_roles}; contrib = {r: [] for r in order_roles}; chk = []
        for t in range(1, 9):
            grad = JS[:, t, :, :] @ WH[g]        # (B, input) d (W_g . h(t)) / d x_s
            grad[:, t + 1] += WI[g]
            pre = P[:, t + 1, g]
            chk.append(np.abs((grad * X).sum(1) - pre).max())
            new = V["new"][:, t].astype(bool); on5 = H[:, t + 1, 5] > 0
            m = {"record": new, "not record, gate 5 on": ~new & on5, "not record, gate 5 off, x>0": ~new & ~on5 & (X[:, t] > 0)}[case]
            # roles of earlier numbers
            xs = X[:, :t]
            m1, m2 = V["m1"][:, t], V["m2"][:, t]
            is_m1 = (xs == m1[:, None]) & (m1[:, None] > 0)
            is_m2 = (xs == m2[:, None]) & (m2[:, None] > 0)
            rec = V["new"][:, :t].astype(bool) & (xs > 0)
            roles = {"m1": is_m1, "m2": is_m2 & ~is_m1, "other record": rec & ~is_m1 & ~is_m2,
                     "kept": (xs > 0) & ~rec & ~is_m1 & ~is_m2 & keep[:, :t],
                     "forgotten": (xs > 0) & ~rec & ~is_m1 & ~is_m2 & ~keep[:, :t], "negative": xs <= 0}
            for r, rm in roles.items():
                sel = rm & m[:, None]
                acc[r].append(grad[:, :t][sel]); contrib[r].append(np.abs(grad[:, :t] * xs)[sel])
            acc["x_t"].append(grad[m, t]); contrib["x_t"].append(np.abs(grad[m, t] * X[m, t]))
            acc["next x"].append(grad[m, t + 1]); contrib["next x"].append(np.abs(grad[m, t + 1] * X[m, t + 1]))
        tot = sum(np.concatenate(contrib[r]).sum() for r in order_roles)
        print(f"  {case}: max |sum g x - pre| = {max(chk):.1e}")
        for r in order_roles:
            a = np.concatenate(acc[r]); c = np.concatenate(contrib[r])
            if len(a):
                print(f"    {r:13s} n={len(a):7d}  g mean {a.mean():7.2f}  sd {a.std():6.2f}  median |g| {np.median(np.abs(a)):6.2f}  share of sum |g x| {c.sum() / tot:5.3f}")
