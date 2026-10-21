"""ARC's nudge tests, run on v23 (one-way memory, deficit sign). See README step 7 for the predictions."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import DEV, dt, T, ARC, top3, near_lists
from algzoo.handbuilt_v23 import build_v23
net = build_v23(); W = T(net.params); I = net.idx
MEM = [I[f"mem{j}"] for j in range(8)]; HIDE = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]; KEEP = [I["keep"]]
GROUPS_ARC = {"memory": [0, 3, 10, 11, 12, 13, 14, 15], "hiding": [1, 6, 7, 8], "keep": [5]}
def run(W, X, nudges=(), steps=10, extra=0):
    h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt)
    for t in range(steps + extra):
        x = X[:, t:t + 1] if t < steps else torch.zeros(len(X), 1, device=DEV, dtype=dt)
        h = torch.relu(h @ W[1].T + x * W[0])
        for g, st, f in nudges:
            if t in st: h = h.clone(); h[:, g] = h[:, g] * f
    return h @ W[2].T
def lists(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); return torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
if __name__ == "__main__":
    pct = lambda m: f"{100*m.double().mean().item():5.1f}%"
    # 1. memory turned up / down at step 5 or 7
    X = near_lists(2_000_000, seed=3); l, s, t3 = top3(X)
    for T_ in (5, 7):
        for who, a, b in (("second early, third late", s, t3), ("third early, second late", t3, s)):
            m = (l <= T_ - 3) & (a >= l + 2) & (a <= T_ - 2) & (b >= T_ + 1); Xm, am, lm = X[m], a[m], l[m]
            for name, Wn, mem in (("v23", W, MEM), ("ARC", ARC, GROUPS_ARC["memory"])):
                row = [f"{e:+.2f}: {pct(run(Wn, Xm, [(mem, [T_], 1 + e)]).argmax(1) == am)}" for e in (-0.05, 0.05)]
                print(f"1. memory nudged at step {T_}, {who}, {name}: earlier wins, base {pct(run(Wn, Xm).argmax(1) == am)} | " + " | ".join(row), flush=True)
    # 2. keep gate scaled at every step: near-tie wins of a last-step contender vs others
    for name, Wn, kp in (("v23", W, KEEP), ("ARC", ARC, GROUPS_ARC["keep"])):
        for who, a in (("second at step 9", s), ("third at step 9", t3)):
            m = a == 9; Xm = X[m]; sm = s[m]
            row = [f"{f:.2f}: {pct(run(Wn, Xm, [(kp, range(10), f)]).argmax(1) == 9)}" for f in (0.95, 1.05)]
            print(f"2. keep gate scaled, {name}, {who}: step-9 contender wins, base {pct(run(Wn, Xm).argmax(1) == 9)} | " + " | ".join(row), flush=True)
        m = (s < 9) & (t3 < 9); Xm, sm = X[m], s[m]
        row = [f"{f:.2f}: {pct(run(Wn, Xm, [(kp, range(10), f)]).argmax(1) == sm)}" for f in (0.95, 1.05)]
        print(f"2. keep gate scaled, {name}, contenders at steps 0-8: second wins, base {pct(run(Wn, Xm).argmax(1) == sm)} | " + " | ".join(row), flush=True)
    # 3. one extra step with input 0
    Xr = lists(500_000, 11); _, sr, _ = top3(Xr); ok = sr >= 1
    for name, Wn in (("v23", W), ("ARC", ARC)):
        a = run(Wn, Xr, extra=1).argmax(1)
        print(f"3. extra step with input 0, {name}: names the position before the second {pct((a == sr - 1)[ok])} (second at 1-9); names the second {pct((a == sr)[ok])}", flush=True)
    # 4. hiding turned down at step 5, largest arrived by step 3, on random lists
    ml = (top3(Xr)[0] <= 3); Xm = Xr[ml]; lm = top3(Xm)[0]
    for name, Wn, hd in (("v23", W, HIDE), ("ARC", ARC, GROUPS_ARC["hiding"])):
        row = [f"x{f:.1f}: {pct(run(Wn, Xm, [(hd, [5], f)]).argmax(1) == lm)}" for f in (0.9, 0.8)]
        print(f"4. hiding turned down at step 5, {name}: largest named, base {pct(run(Wn, Xm).argmax(1) == lm)} | " + " | ".join(row), flush=True)
