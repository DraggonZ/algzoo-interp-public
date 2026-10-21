"""Where does the mean-background reduction's bias come from? On the same random lists (development check, sampling
used only to look), compare the network's answer on the real list with its answer on the list's reduction (the three
largest kept, the other seven replaced by the mean of a normal truncated below the third). Paired, so the difference
of the two accuracies has a small standard error. Split the disagreements by region: second-largest below 0; a < 0.1
(near the 2nd-3rd tie) ; b < 0.1 (near the 1st-2nd tie) ; both ; neither."""
import sys, numpy as np, torch
import common as c
log = c.Log("bias_where.log")
names = sys.argv[1:] or ["ARC", "200", "208", "212"]
N, CH = 10_000_000, 1_000_000
for name, W in c.nets():
    if name not in names: continue
    g = torch.Generator(device=c.DEV).manual_seed(5); acc = torch.zeros(5, 2, device=c.DEV, dtype=c.dt); nr = 0; nn = 0
    for i in range(0, N, CH):
        X = torch.randn(CH, 10, device=c.DEV, dtype=c.dt, generator=g)
        top = X.topk(3, 1); pos = top.indices; val = top.values
        S, _ = c.scores(W, X); right = S.argmax(1) == pos[:, 1]
        S2, _ = c.scores(W, c.inputs(pos, val)); rright = S2.argmax(1) == pos[:, 1]
        a = val[:, 1] - val[:, 2]; b = val[:, 0] - val[:, 1]
        reg = torch.where(val[:, 1] < 0, 0, torch.where((a < .1) & (b >= .1), 1, torch.where((b < .1) & (a >= .1), 2, torch.where((a < .1) & (b < .1), 3, 4))))
        for r in range(5):
            m = reg == r
            acc[r, 0] += (m & rright & ~right).sum(); acc[r, 1] += (m & right & ~rright).sum()
        nr += rright.sum().item(); nn += right.sum().item()
    acc = (100 * acc / N).cpu().numpy(); d = 100 * (nr - nn) / N
    names_r = ["m2<0", "a<0.1 only", "b<0.1 only", "both<0.1", "neither"]
    log(f"{name:>4s}: network {100*nn/N:.3f}%, reduction on the same lists {100*nr/N:.3f}%, difference {d:+.3f} points")
    log(f"      lists where the two disagree: {acc[:,0].sum()+acc[:,1].sum():.2f}% ({acc[:,0].sum():.2f}% reduction right & network wrong, {acc[:,1].sum():.2f}% the reverse)")
    log("      reduction right & network wrong / network right & reduction wrong (points): " + "; ".join(f"{k} {acc[r,0]:.3f}/{acc[r,1]:.3f}" for r, k in enumerate(names_r)))
