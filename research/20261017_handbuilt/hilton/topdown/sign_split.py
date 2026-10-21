"""ARC's network, v18 and the other trained networks split by the signs of the largest and second-largest. Per group:
probability share, accuracy, share of each network's mistakes that land in the group, ARC's mistakes reproduced, and
what ARC and v18 name when wrong (largest, third, position 0 among others, other). Grid 40 x 48 x 48."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v18 import build_v18
from mistakes_other_nets import T, net
DEV = "cuda"; dt = torch.float64
if __name__ == "__main__":
    v18 = net(T(build_v18().params))
    tr = [net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))) for s in (103, 111, 115, 209, 210)]
    nV = len(sl.wtt); GR = ["L+ S+", "L+ S-", "L- S-"]; st = {}
    def add(key, v):
        st[key] = st.get(key, 0.0) + v
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]
        a = {"ARC": sl.run(X).argmax(1), "v18": v18(X).argmax(1)}
        at = [f(X).argmax(1) for f in tr]
        grp = torch.where(val[:, 1] > 0, 0, torch.where(val[:, 0] > 0, 1, 2))
        for gi in range(3):
            m = grp == gi; add((gi, "p"), (w * m).sum().item())
            for nm, an in list(a.items()) + [("trained", None)]:
                if an is None:
                    add((gi, nm, "right"), sum((w * m * (x == y)).sum().item() for x in at) / len(at))
                    add((gi, nm, "repro"), sum((w * m * (x == a["ARC"]) * (a["ARC"] != y)).sum().item() for x in at) / len(at)); continue
                wr = m & (an != y); add((gi, nm, "right"), (w * m * (an == y)).sum().item())
                kind = torch.where(an == pos[:, 0], 0, torch.where(an == pos[:, 2], 1, torch.where(an == 0, 2, 3)))
                for k in range(4): add((gi, nm, "kind", k), (w * wr * (kind == k)).sum().item())
                if nm == "v18": add((gi, nm, "repro"), (w * wr * (an == a["ARC"])).sum().item())
    P = sum(st[(g, "p")] for g in range(3)); werr = {nm: sum(st[(g, nm, "p") if False else (g, "p")] - st[(g, nm, "right")] for g in range(3)) for nm in ("ARC", "v18", "trained")}
    for gi, gn in enumerate(GR):
        p = st[(gi, "p")]; print(f"== {gn}: {100*p/P:.1f}% of lists")
        for nm in ("ARC", "v18", "trained"):
            e = p - st[(gi, nm, "right")]
            line = f"  {nm:8s} accuracy {100*st[(gi, nm, 'right')]/p:6.2f}%  holds {100*e/werr[nm]:5.1f}% of its mistakes"
            if nm != "ARC": line += f"  ARC mistakes reproduced {100*st[(gi, nm, 'repro')]/(p - st[(gi, 'ARC', 'right')]):5.1f}%"
            if nm != "trained": line += "  wrong answers: " + ", ".join(f"{k} {100*st[(gi, nm, 'kind', j)]/max(e,1e-30):.0f}%" for j, k in enumerate(("largest", "third", "pos 0 (other)", "other")))
            print(line)
