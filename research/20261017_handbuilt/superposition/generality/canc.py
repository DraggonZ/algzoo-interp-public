"""Question 3: cancellation per target network. The final state h times the readout difference gives each neuron's push
on a decision; the decision is the sum. Ratio = median over lists of (sum of |pushes|) / |decision|, per neuron (as
an earlier check) and per group (pushes summed within a group first). Groups: the hiding group
found per network (neurons a new largest switches off, common_gen.hiding_group) against all the others; for ARC also its
named roles (memory 0,3,10-15; hiding 1,6,7,8; gates 5,9; store and detector 2,4). Decisions: second minus third, and
second minus largest. Lists: random (typical), random with the second positive, near ties (third within 5% of the second,
second positive)."""
from common_gen import *
ARC_ROLES = [[0, 3, 10, 11, 12, 13, 14, 15], [1, 6, 7, 8], [5, 9], [2, 4]]
def ratios(W, X, groups):
    L, S, Tq = top3(X)
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    out = []
    for other in (Tq, L):
        P = h * (W[2][S] - W[2][other]); D = P.sum(1).abs().clamp_min(1e-9)
        row = [(P.abs().sum(1) / D).median().item()]
        for gs in groups: row.append((torch.stack([P[:, g].sum(1) for g in gs], 1).abs().sum(1) / D).median().item())
        out.append(row)
    return out
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(5); Xr = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    Xp = Xr[Xr.sort(1, descending=True).values[:, 1] > 0]; Xn = near_lists(100000, seed=3)
    print("Each cell: median of (sum of |pushes|) / |decision|. 'neuron' = per neuron; 'hide/rest' = two groups (the hiding group, the rest); 'roles' = ARC's four named roles.\n")
    for nm, W in TARGETS.items():
        hid, _, _ = hiding_group(W); rest = [j for j in range(16) if j not in hid]; groups = [[hid, rest]]
        if nm == "ARC": groups.append(ARC_ROLES)
        print(f"{nm} (hiding group {hid})")
        for lab, X in (("random lists", Xr), ("random, second > 0", Xp), ("near ties", Xn)):
            r = ratios(W, X, groups)
            for dn, row in zip(("second vs third", "second vs largest"), r):
                print(f"  {lab:20s} {dn:18s} neuron {row[0]:7.1f}   hide/rest {row[1]:6.1f}" + (f"   roles {row[2]:6.1f}" if len(row) > 2 else ""))
