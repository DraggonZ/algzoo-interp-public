"""Where is the network wrong, in the converged mean-background estimate? Split the 4.7% error mass by the two gaps:
near the 2nd-3rd tie (a < 0.1), near the 1st-2nd tie (b < 0.1), both, or neither (the 'bulk')."""
import numpy as np, torch
import full_mean, adf
W = adf.net(None); V, wt = full_mean.grid(56, 96); Vt = torch.tensor(V, device="cuda", dtype=torch.float64); wt = torch.tensor(wt, device="cuda", dtype=torch.float64)
a = Vt[:, 1] - Vt[:, 2]; b = Vt[:, 0] - Vt[:, 1]; u = Vt[:, 2]
cls = {k: torch.zeros((), device="cuda", dtype=torch.float64) for k in ("a<0.1 only", "b<0.1 only", "both", "neither")}
mass = 0
for i in range(0, 720, 8):
    tt = full_mean.T[i:i + 8]; nV = len(wt); pos = tt.repeat_interleave(nV, 0); val = Vt.repeat(len(tt), 1)
    wrong = 1 - full_mean.win_hard(W, pos, val); w = wt.repeat(len(tt)) * wrong
    A, B = a.repeat(len(tt)), b.repeat(len(tt))
    cls["a<0.1 only"] += (w * ((A < .1) & (B >= .1))).sum(); cls["b<0.1 only"] += (w * ((B < .1) & (A >= .1))).sum()
    cls["both"] += (w * ((A < .1) & (B < .1))).sum(); cls["neither"] += (w * ((A >= .1) & (B >= .1))).sum()
tot = wt.sum() * 720
for k, v in cls.items(): print(f"error mass {k:12s}: {100 * (v / tot).item():.2f} points")
for thr in (0.2, 0.3, 0.5):
    print("(see also larger thresholds by rerunning)") if thr == 0.2 else None
