"""Weights from the sharper rule (README experiment 22). L1's input values (run on its own outputs, fit lists) are the
target; each hiding neuron's 16 incoming recurrent weights are fitted on ARC's previous states (squared error where
the rule's value > 0, hinge max(0, pre)^2 where < 0); the input weight is ARC's. Compared with ARC; substitution cost on
the usual 2^22 lists (paired)."""
import io, contextlib
import numpy as np
import torch as th
from harness import *  # noqa
import step3_recursive as s3
import step5_lean as s5
from algzoo.core import example

if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        feat = s5.make(["sd", "xprev", "lategate"])
        co, _ = s3.fit_group(feat, "L1")
    WI, WH = simlib.WI, simlib.WH
    # rule run on its own outputs on the fit lists: pre-activation values
    Ff = {n: feat(Xf, n) for n in HID}
    rule_pre = np.zeros((len(Xf), 10, 4)); prev = np.zeros((len(Xf), 4))
    for t in range(10):
        for i, n in enumerate(HID):
            rule_pre[:, t, i] = np.concatenate([Ff[n][:, t], prev], -1) @ co[n]
        prev = np.maximum(rule_pre[:, t], 0)
    Hprev = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1).reshape(-1, 16)
    fitted = {}
    for i, n in enumerate(HID):
        tgt = th.tensor(rule_pre[:, :, i].reshape(-1)); Hp = th.tensor(Hprev); xin = th.tensor(Xf.reshape(-1)) * WI[n]
        on = tgt > 0
        w = th.tensor(WH[n] * 0, requires_grad=True)
        opt = th.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); pre = Hp @ w + xin
            L = th.where(on, (pre - tgt) ** 2, th.relu(pre) ** 2).mean(); L.backward(); return L
        for _ in range(4):
            opt.step(closure)
        fitted[n] = w.detach().numpy()
        a = WH[n]; f = fitted[n]
        print(f"neuron {n}: cosine with ARC {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance {np.linalg.norm(f - a) / np.linalg.norm(a):.2f}")
        print("   ARC:    " + " ".join(f"{v:+.2f}" for v in a))
        print("   fitted: " + " ".join(f"{v:+.2f}" for v in f))

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
    for rows in [[1], [6], [7], [8], HID]:
        P = [w.copy() for w in W0]
        for n in rows:
            P[1][n] = fitted[n]
        l = run_g(P)
        print(f"rows {rows} from the rule: accuracy change {100 * ((l.argmax(-1) == Y).float().mean().item() - acc0):+.2f} points, KL {(pb * (lb - l)).sum(-1).mean().item():.3f}")
