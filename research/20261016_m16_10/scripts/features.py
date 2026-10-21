"""Candidate homogeneous features of a prefix x_0..x_{t-1}, for regressing neurons."""
import torch as th


def prefix_features(x, t):
    """x: [N, n]; returns dict name -> [N] tensor for the prefix of length t."""
    N = x.shape[0]
    z = th.zeros(N, 1, device=x.device, dtype=x.dtype)
    pre = x[:, :t]
    f = {}
    withz = th.cat([z, pre], 1)
    s = withz.sort(1, descending=True).values
    f["max0"] = s[:, 0]                        # max(0, x_0..x_{t-1})
    f["2nd0"] = s[:, 1]                        # 2nd largest of (0, x...)
    s2 = pre.sort(1, descending=True).values
    f["max"] = s2[:, 0]
    if t >= 2:
        f["2nd"] = s2[:, 1]
    if t >= 3:
        f["3rd"] = s2[:, 2]
    for k in range(1, min(t, 4) + 1):
        f[f"x[t-{k}]"] = pre[:, t - k]
        f[f"relu(x[t-{k}])"] = pre[:, t - k].clamp(min=0)
        # leave-one-out max (with 0) excluding x_{t-k}
        keep = [s_ for s_ in range(t) if s_ != t - k]
        lo = th.cat([z, pre[:, keep]], 1).max(1).values
        f[f"loo0[t-{k}]"] = lo
    if t >= 2:
        f["max0_prev"] = th.cat([z, pre[:, : t - 1]], 1).max(1).values
    return f


def lag_features(x, t, maxlag=10):
    """Value-times-one-hot features for the lag of the max and of the 2nd max (no zero)."""
    pre = x[:, :t]
    order = pre.argsort(1, descending=True)
    vals = pre.gather(1, order)
    f = {}
    lag1 = t - 1 - order[:, 0]
    for L in range(t):
        f[f"max*[lag1={L}]"] = vals[:, 0].clamp(min=0) * (lag1 == L)
    if t >= 2:
        lag2 = t - 1 - order[:, 1]
        for L in range(t):
            f[f"2nd*[lag2={L}]"] = vals[:, 1].clamp(min=0) * (lag2 == L)
    return f
