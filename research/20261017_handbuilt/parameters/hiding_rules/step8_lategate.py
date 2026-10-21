"""Variant of L1 with the late gate's signal as the account describes it (README experiment 22, follow-up): one step
after a new largest, about 1.5 x the new largest; here lag(newmax * x) (the previous number if it was a new largest),
in place of, or beside, the previous jump. Activity errors (as step3) and weights from the rule (as step7)."""
import io, contextlib, sys
import numpy as np
import torch as th
from harness import *  # noqa
import step3_recursive as s3
import step4_ablate as s4
from step3_recursive import account_features
from algzoo.core import example

WI, WH = simlib.WI, simlib.WH


def make(extra):
    def f(X, n):
        V, P = s4.parts(X)
        F = [account_features(X, n)[..., k] for k in range(5)] + P["sd"] + P["xprev"]
        if "jump" in extra:
            F += P["lategate"]
        if "newmax" in extra:
            F += [lag(V["newmax"] * V["x"])]
        return np.stack(F, -1)
    return f


dev = "cuda"; N = 2**22
W0 = [np.asarray(p, float) for p in example()]
Xg = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev); Y = Xg.argsort(-1)[:, -2]


@th.no_grad()
def run_g(P):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ b.T + Xg[:, t, None] * a)
    return th.log_softmax(h @ c.T, -1)


lb = run_g(W0); pb = lb.exp(); acc0 = (lb.argmax(-1) == Y).float().mean().item()


def weights_from_rule(feat, co):
    Ff = {n: feat(Xf, n) for n in HID}
    rule_pre = np.zeros((len(Xf), 10, 4)); prev = np.zeros((len(Xf), 4))
    for t in range(10):
        for i, n in enumerate(HID):
            rule_pre[:, t, i] = np.concatenate([Ff[n][:, t], prev], -1) @ co[n]
        prev = np.maximum(rule_pre[:, t], 0)
    Hp = th.tensor(np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1).reshape(-1, 16))
    fitted = {}
    for i, n in enumerate(HID):
        tgt = th.tensor(rule_pre[:, :, i].reshape(-1)); xin = th.tensor(Xf.reshape(-1)) * WI[n]; on = tgt > 0
        w = th.zeros(16, dtype=th.float64, requires_grad=True)
        opt = th.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); pre = Hp @ w + xin
            L = th.where(on, (pre - tgt) ** 2, th.relu(pre) ** 2).mean(); L.backward(); return L
        for _ in range(4):
            opt.step(closure)
        fitted[n] = w.detach().numpy()
    return fitted


if __name__ == "__main__":
  for lab, extra in [("L1 with the late gate as lag(newmax x)", ["newmax"]), ("L1 with both late-gate signals", ["jump", "newmax"])]:
      feat = make(extra)
      co, _ = s3.fit_group(feat, lab)
      fitted = weights_from_rule(feat, co)
      for n in HID:
          a, f = WH[n], fitted[n]
          print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance "
                f"{np.linalg.norm(f - a) / np.linalg.norm(a):.2f}; from late gate ARC {a[9]:+.2f} fitted {f[9]:+.2f}")
      for rows in [[1], [6], [7], [8], HID]:
          P = [w.copy() for w in W0]
          for n in rows:
              P[1][n] = fitted[n]
          l = run_g(P)
          print(f"  rows {rows} from the rule: accuracy {100 * ((l.argmax(-1) == Y).float().mean().item() - acc0):+.2f} points, KL {(pb * (lb - l)).sum(-1).mean().item():.3f}")
