"""Step 2: one row of quantities per network (definitions in README.md). Every quantity is unchanged by rescaling a
neuron (weights compared in the activity gauge of nets.gauge, or as ratios within one neuron's row and column) and by
neuron order (neurons are named by the roles of roles.py). Writes quantities.csv and quantities_detail.json."""
import json, itertools, sys, csv
from collections import Counter
import numpy as np, torch
from nets import *
from roles import find_roles

def task_level(Wt, n_chunks=4, chunk=2 ** 20, seed=11):
    tot = right = 0; kinds = Counter(); nt = nt_right = nt_third = 0
    for c in range(n_chunks):
        X = lists(chunk, seed + c); S = run(Wt, X)[:, -1] @ Wt[2].T; pred = S.argmax(1)
        o = X.argsort(1, descending=True); first, second, third = o[:, 0], o[:, 1], o[:, 2]
        ok = pred == second; tot += len(X); right += ok.sum().item()
        few = (X > 0).sum(1) <= 1; w = ~ok
        kinds["few positives"] += (w & few).sum().item(); kinds["third"] += (w & ~few & (pred == third)).sum().item()
        kinds["largest"] += (w & ~few & (pred == first)).sum().item()
        kinds["fourth or lower"] += (w & ~few & (pred != third) & (pred != first)).sum().item()
        v = X.sort(1, descending=True).values; near = ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1].abs()) & (v[:, 1] > 0)
        nt += near.sum().item(); nt_right += (near & ok).sum().item(); nt_third += (near & (pred == third)).sum().item()
    nw = tot - right
    return {"accuracy": right / tot, **{f"mistakes: {k}": kinds[k] / nw for k in ["third", "largest", "fourth or lower", "few positives"]},
            "near ties: accuracy": nt_right / nt, "near ties: third wins": nt_third / nt}

def ages(X):
    """[n, 10] age of the largest so far after step t (t - its position)."""
    t = torch.arange(10, device=DEV)
    best = torch.zeros(len(X), 10, dtype=torch.long, device=DEV); cur = torch.zeros(len(X), dtype=torch.long, device=DEV)
    for s in range(1, 10):
        cur = torch.where(X[:, s] > X[torch.arange(len(X)), cur], torch.full_like(cur, s), cur); best[:, s] = cur
    return t[None, :] - best

def mechanism(W, R, n=200000, seed=7):
    Wt = T(W); X = lists(n, seed); H = run(Wt, X); on = H > 0
    (wi, wh, wo), sig = gauge(W, H)
    q, det = {}, {}
    s, d, hid, mem = R["store"], R["detector"], R["hiding"], R["memory"]; gates = R["late"] + R["keep"]
    # store
    q["store: correlation with largest"] = R["store_corr"]
    A = H[:, 3:, :].reshape(-1, 16); b = X.cummax(1).values[:, 2:9].reshape(-1)
    A1 = torch.cat([A, torch.ones(len(A), 1, device=DEV, dtype=DT)], 1)
    coef = torch.linalg.lstsq(A1, b[:, None]).solution; r2 = 1 - ((A1 @ coef)[:, 0] - b).var() / b.var()
    q["store: largest read from all neurons (R2)"] = r2.item()
    q["store: self-weight (fade per step)"] = W[1][s, s]
    # readout shift relation
    pats = Counter(map(lambda r: r.tobytes(), on[:, 4:9].reshape(-1, 16).cpu().numpy())); top = np.frombuffer(pats.most_common(1)[0][0], dtype=bool)
    M = np.diag(top.astype(float)) @ wh; Wc = wo - wo.mean(0)
    P = Wc[:-1] @ M; P = P - P.mean(0)          # push of rows 0..8; centred the same way
    cosk = [P[k] @ Wc[k + 1] / np.linalg.norm(P[k]) / np.linalg.norm(Wc[k + 1]) for k in range(9)]
    q["readout shift: median cosine (k=0..7)"] = float(np.median(cosk[:8])); det["readout shift cosines"] = np.round(cosk, 3).tolist()
    q["readout shift: share of steps in that pattern"] = pats.most_common(1)[0][1] / (5 * n)
    # memory block spectrum
    ev = np.linalg.eigvals(W[1][np.ix_(mem, mem)]); ang = np.degrees(np.abs(np.angle(ev))); mod = np.abs(ev)
    det["memory eigenvalues (angle, modulus)"] = sorted({(round(a, 1), round(m, 3)) for a, m in zip(ang, mod)})
    rot = (ang > 3) & (ang < 177)
    q["memory: number of rotating pairs"] = int(rot.sum() // 2)
    q["memory: slowest rotation angle"] = float(ang[rot].min()) if rot.any() else np.nan
    q["memory: largest modulus"] = float(mod.max())
    q["memory: median modulus"] = float(np.median(mod))
    # switch-off ages of memory neurons
    AG = ages(X)[:, 1:]; offm = ~on[:, 1:]
    clipped = []
    for j in mem:
        rates = [offm[:, :, j][AG == a].double().mean().item() if (AG == a).any() else np.nan for a in range(10)]
        for a in range(1, 9):
            other = offm[:, :, j][(AG != a)].double().mean().item()
            if rates[a] >= 0.85 and other <= 0.30: clipped.append((j, a, round(rates[a], 2), round(other, 2)))
    det["age-clipped memory neurons (neuron, age, off rate there, elsewhere)"] = clipped
    q["memory: age-clipped neurons"] = len({c[0] for c in clipped})
    q["memory: ages"] = " ".join(str(a) for a in sorted({c[1] for c in clipped})) or "-"
    # blindness of rows outside the gates to the arriving number through the hiding neurons
    if len(hid) >= 2:
        rows = [i for i in range(16) if i not in gates]; bl, nl = [], []
        signs = np.array(list(itertools.product([-1, 1], repeat=len(hid))))
        for i in rows:
            t = W[1][i, hid] * W[0][hid]; tot = np.abs(t).sum()
            bl.append(abs(t.sum()) / tot); nl.append(np.mean(np.abs(signs @ t)) / tot)
        q["blindness: median ratio (0 = blind)"] = float(np.median(bl)); q["blindness: random-sign null"] = float(np.median(nl))
        q["blindness: share of rows below 0.1"] = float(np.mean(np.array(bl) < 0.1))
        det["blindness by row"] = {int(i): round(float(v), 3) for i, v in zip(rows, bl)}
    # hiding comparators: reference = -(other inputs)/input weight, fitted on largest and second-largest so far
    if hid:
        Hp = H[:, 1:9]; Xp = X[:, :9]; srt = torch.stack([Xp[:, :k + 1].sort(1, descending=True).values[:, :2] if k else
              torch.stack([Xp[:, 0], torch.full_like(Xp[:, 0], np.nan)], 1) for k in range(1, 9)], 1)
        Mx, Sx = srt[:, :, 0].reshape(-1), srt[:, :, 1].reshape(-1)
        fits = []
        for j in hid:
            ref = (-(Hp @ Wt[1][j]) / W[0][j]).reshape(-1)
            A3 = torch.stack([Mx, Sx, torch.ones_like(Mx)], 1); c = torch.linalg.lstsq(A3, ref[:, None]).solution[:, 0]
            r2 = 1 - (A3 @ c - ref).var() / ref.var()
            fits.append((j, round(c[0].item(), 2), round(c[1].item(), 2), round(c[2].item(), 2), round(r2.item(), 3), round(abs(W[0][j]) / sig[j], 2)))
        det["hiding references (neuron, x largest, x second, const, R2, gain)"] = fits
        q["hiding: reference fit R2 (median)"] = float(np.median([f[4] for f in fits]))
        q["hiding: reference weight on largest+second (median)"] = float(np.median([f[1] + f[2] for f in fits]))
        q["hiding: share of reference on largest (median)"] = float(np.median([f[1] / (f[1] + f[2]) if f[1] + f[2] else np.nan for f in fits]))
        q["hiding: gain (|input weight| / activity sd), median"] = float(np.median([f[5] for f in fits]))
        q["hiding: input weight sign negative (share)"] = float(np.mean([W[0][j] < 0 for j in hid]))
    # overnight relations
    if d is not None:
        rs, rd = wh[s], wh[d]
        q["(a) detector row vs store row: cosine"] = float(rd @ rs / np.linalg.norm(rd) / np.linalg.norm(rs))
        cfit = -(rd @ rs) / (rs @ rs); cpred = wh[s, s] / wh[s, d]
        q["(a) c fitted / c from overwrite"] = float(cfit / cpred)
        q["(a) detector loose"] = int(R["detector_loose"])
        if hid:
            vd = np.r_[wi[d], wh[d]]
            cs = {j: float(np.r_[wi[j], wh[j]] @ vd / np.linalg.norm(np.r_[wi[j], wh[j]]) / np.linalg.norm(vd)) for j in hid}
            jm = min(cs, key=cs.get); q["(b) mirror: most negative cosine with detector"] = cs[jm]
            det["(b) hiding cosines with detector"] = cs
            qs = [(W[1][j, s] / W[0][j]) / (W[1][d, s] / W[0][d]) for j in hid]
            q["(c) hiding store ratio / detector's (median)"] = float(np.median(qs))
            q["(c) share of hiding neurons within 0.7-1.3"] = float(np.mean([(0.7 <= v <= 1.3) for v in qs]))
            det["(c) ratios"] = np.round(qs, 3).tolist()
        q["(c') detector store weight / input weight x store activity sd"] = float(W[1][d, s] / W[0][d] * sig[s])
    shares = []
    for i in mem:
        row = wh[i, mem] ** 2; tot = row.sum()
        for k, j in enumerate(mem):
            if j != i: shares.append((row[k] / tot, i, j))
    best = max(shares); q["(d) delay line: largest one-source share in a memory row"] = float(best[0])
    det["(d) best delay pair (share, row, source)"] = (round(best[0], 3), best[1], best[2])
    q["(d) memory rows with one source above 0.8"] = sum(1 for i in mem if max(v for v, a, b in shares if a == i) > 0.8)
    cr = np.abs(W[2].mean(0)) / W[2].std(0); q["(e) readout column |mean| / sd (rms)"] = float(np.sqrt((cr ** 2).mean()))
    return q, det

if __name__ == "__main__":
    nets = networks(); roles = json.load(open("roles.json")); rows, details = [], {}
    only = sys.argv[1:]
    for lab, W in nets.items():
        if only and lab not in only: continue
        R = roles[lab]; tq = task_level(T(W)); mq, dd = mechanism(W, R)
        row = {"net": lab, "hiding size": len(R["hiding"]), "memory size": len(R["memory"]), "late gate": len(R["late"]),
               "keep gate": len(R["keep"]), "strict detector": len(R["detectors"]), **tq, **mq}
        rows.append(row); details[lab] = dd
        print(lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "net"}, flush=True)
        if only: print(json.dumps(dd, indent=1, default=str))
    if not only:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open("quantities.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
        json.dump(details, open("quantities_detail.json", "w"), indent=1, default=str)
