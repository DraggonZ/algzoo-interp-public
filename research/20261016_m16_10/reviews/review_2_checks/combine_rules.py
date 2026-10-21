"""Review 2 check (sampling-based diagnostic, not an estimator): which way of combining single-low-input
effects, with one constant from the base model, tracks the 96 copies?

Per sampled input, L0 = logits with only the top two kept; L_p = logits with the top two plus low input p.
  add      : argmax(L0 + sum_p (L_p - L0)) == p2                      (run-2 additive surrogate)
  add_late : same, summing only low inputs after the last spike (earlier ones zero)
  add_early: same, only low inputs before the last spike
  shrink   : argmax(L0 + c * sum_p (L_p - L0)) == p2, c set on the base model to give 0.9531
  indep    : two-spike answer right AND no single low input alone makes it wrong. Given positions and
             (m1, m2) the low inputs are independent, so its mean is exactly
             E[ 1(Z2 right) * prod_p (1 - f_p(m1, m2)) ]: a product rule with no convolution.
Ratio calibration: estimate * (0.9531 / base value). Nothing is fitted to the copies.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

dev = "cuda"
N = 2**18
BASE_ACC = 0.9531


@th.no_grad()
def parts(params, seed):
    g = th.Generator(device=dev).manual_seed(seed)
    x = th.randn(N, 10, generator=g, device=dev, dtype=th.float64)
    v, idx = x.sort(-1, descending=True)
    p2 = idx[:, 1]
    last = idx[:, :2].max(1).values
    top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
    L0 = forward(params, top2)
    D = th.zeros(8, N, 10, dtype=x.dtype, device=dev)
    late = th.zeros(8, N, dtype=th.bool, device=dev)
    for k in range(8):
        xs = top2.clone().scatter_(1, idx[:, 2 + k:3 + k], v[:, 2 + k:3 + k])
        D[k] = forward(params, xs) - L0
        late[k] = idx[:, 2 + k] > last
    full = (forward(params, x).argmax(-1) == p2).double().mean().item()
    return L0, D, late, p2, full


def acc_of(L, p2):
    return (L.argmax(-1) == p2).double().mean().item()


def rules(L0, D, late, p2, c):
    r = {}
    r["add"] = acc_of(L0 + D.sum(0), p2)
    r["add_late"] = acc_of(L0 + (D * late[..., None]).sum(0), p2)
    r["add_early"] = acc_of(L0 + (D * (~late)[..., None]).sum(0), p2)
    r["shrink"] = acc_of(L0 + c * D.sum(0), p2)
    z2 = L0.argmax(-1) == p2
    single_ok = ((L0[None] + D).argmax(-1) == p2[None]).all(0)
    r["indep"] = (z2 & single_ok).double().mean().item()
    r["Z2"] = z2.double().mean().item()
    return r


def main():
    L0, D, late, p2, full = parts(example(), 0)
    lo, hi = 0.0, 1.0
    for _ in range(30):                        # bisection for c on the base model
        c = 0.5 * (lo + hi)
        if acc_of(L0 + c * D.sum(0), p2) > BASE_ACC:
            lo = c
        else:
            hi = c
    C = 0.5 * (lo + hi)
    base = rules(L0, D, late, p2, C)
    print(f"base: c = {C:.4f}; " + ", ".join(f"{k} {v:.4f}" for k, v in base.items()) + f"; full sampled {full:.4f}")
    del L0, D

    names, acc, models = load_bench(("mult",))
    sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
    keys = list(base)
    R = []
    for i, m in enumerate(models):
        L0, D, late, p2, full = parts(m, 1 + i)
        r = rules(L0, D, late, p2, C)
        R.append([r[k] for k in keys])
        del L0, D
    R = np.array(R)
    np.savez("combine_rules.npz", R=R, keys=keys, acc=acc, base=[base[k] for k in keys], c=C)
    t = acc
    for j, k in enumerate(keys):
        for lab, e in ((k, R[:, j]), (k + " x base ratio", R[:, j] * BASE_ACC / base[k])):
            out = [f"{lab:24s} RMSE {np.sqrt(np.mean((e - t) ** 2)):.4f} bias {np.mean(e - t):+.4f} corr {np.corrcoef(e, t)[0, 1]:.3f}"]
            for s in (0.005, 0.01, 0.02):
                q = sig == s
                out.append(f"σ={s}: {np.sqrt(np.mean((e[q] - t[q]) ** 2)):.4f} ({np.mean(e[q] - t[q]):+.3f})")
            print(" | ".join(out))


if __name__ == "__main__":
    main()
