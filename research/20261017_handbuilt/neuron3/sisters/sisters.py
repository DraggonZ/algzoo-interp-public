"""Neuron-3-like memory neurons in ARC and its 37 sister networks (method and predictions in README.md).
Writes sisters.log (per network), sisters.csv (one row per memory neuron), pairs.csv (every hiding pair x memory neuron).
Numpy only. Fits are analysis fits to each network's own activity; accuracy is only measured."""
import os, sys, json, csv
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
UNI = os.path.join(HERE, "..", "..", "parameters", "universal")
sys.path.insert(0, UNI)
import numpy as np
from nets import networks   # loads the 38 networks (ARC first)

N_LISTS = 200_000
ROLES = json.load(open(os.path.join(UNI, "roles.json")))
DETAIL = json.load(open(os.path.join(UNI, "quantities_detail.json")))
REFKEY = "hiding references (neuron, x largest, x second, const, R2, gain)"


def lists(seed, n=N_LISTS): return np.random.default_rng(seed).standard_normal((n, 10))


def run(W, X):
    wi, wh, wo = W; h = np.zeros((len(X), len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0); P.append(pre); H.append(h)
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def task(X):
    """age: t - position of the largest so far (-1 while no positive number), as memlib.task.
    m1, m2: true largest and second-largest so far (m2 = m1 at step 0, so the gap is 0 there)."""
    n, T = X.shape; age = np.full((n, T), -1); m1 = np.zeros((n, T)); m2 = np.zeros((n, T))
    best = np.zeros(n); arg = np.full(n, -1)
    for t in range(T):
        new = X[:, t] > best; best = np.where(new, X[:, t], best); arg = np.where(new, t, arg)
        age[:, t] = np.where(arg >= 0, t - arg, -1)
        s = -np.sort(-X[:, :t + 1], 1); m1[:, t] = s[:, 0]; m2[:, t] = s[:, 1] if t else s[:, 0]
    return dict(age=age, m1=m1, m2=m2, gap=m1 - m2)


def accuracy(S, X): return float((S.argmax(1) == X.argsort(1)[:, -2]).mean() * 100)


def gap_cells(Hh, V, X):
    """Per (step 2-9, age) cell with a positive largest and >= 2000 lists: covariance of the hiding activities (Cy) and of
    their residuals after least squares on [gap, 1] (Cr). Any linear combination a has cell R2 = 1 - a'Cr a / a'Cy a."""
    out = []
    for t in range(2, 10):
        for a in range(0, t + 1):
            m = (V["age"][:, t] == a)
            if m.sum() < 2000: continue
            Y = Hh[m, t]; A = np.c_[V["gap"][m, t], np.ones(m.sum())]
            c, *_ = np.linalg.lstsq(A, Y, rcond=None); R = Y - A @ c
            A2 = np.c_[A, X[m, t]]; c2, *_ = np.linalg.lstsq(A2, Y, rcond=None); R2 = Y - A2 @ c2   # gap plus the arriving number
            k = Y.shape[1]
            out.append((t, a, np.cov(Y.T).reshape(k, -1), np.cov(R.T).reshape(k, -1), np.cov(R2.T).reshape(k, -1)))
    return out


def median_r2(cells, a, overall_var, with_x=False):
    r = [1 - a @ (Cr2 if with_x else Cr) @ a / (a @ Cy @ a) for _, _, Cy, Cr, Cr2 in cells if a @ Cy @ a > 0.01 * overall_var]
    return (float(np.median(r)) if r else np.nan), len(r)


def cell_key(V, t): return V["age"][:, t - 1]  # cell of the difference written at step t-1, read at step t


def fit_sub(D, V, kind):
    """D[:, t] = difference entering at step t (from h at t-1). Per (t, age at t-1): mean, or [gap at t-1, 1] fit."""
    C = {}
    for t in range(1, 10):
        a = cell_key(V, t)
        for ag in np.unique(a):
            m = a == ag
            F = np.ones((m.sum(), 1)) if kind == "mean" else np.c_[V["gap"][m, t - 1], np.ones(m.sum())]
            C[(t, ag)] = np.linalg.lstsq(F, D[m, t], rcond=None)[0]
    return C


def run_sub(W, X, V, i, js, C, kind):
    wi, wh, wo = W; h = np.zeros((len(X), len(wi)))
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if t >= 1:
            D = h[:, js] @ wh[i, js]; a = cell_key(V, t); rep = np.zeros(len(X))
            for ag in np.unique(a):
                m = a == ag; c = C.get((t, ag))
                if c is None: rep[m] = D[m]; continue
                rep[m] = c[0] if kind == "mean" else V["gap"][m, t - 1] * c[0] + c[1]
            pre[:, i] += rep - D
        h = np.maximum(pre, 0)
    return h @ wo.T


def analyse(lab, W, rng_pick):
    R = ROLES[lab]; hid, mem = R["hiding"], R["memory"]; wi, wh, wo = W
    Xf, Xt = lists(21), lists(22); Vf, Vt = task(Xf), task(Xt)
    _, P, H = run(W, Xf); base = accuracy(run(W, Xt)[0], Xt)
    sig = H.reshape(-1, H.shape[-1]).std(0); sig = np.where(sig > 1e-12, sig, 1.0)
    rows, prs = [], []
    cells = gap_cells(H[:, :, hid], Vf, Xf) if len(hid) >= 1 else []
    Hall = H[:, 2:10][:, :, hid].reshape(-1, len(hid)) if hid else None
    Cov_all = np.cov(Hall.T).reshape(len(hid), -1) if hid else None
    for i in mem:
        # terms entering pre_i at steps 3..9
        Hp = H[:, 2:9]; terms = Hp * wh[i][None, None, :]                        # [n, 7, 16]
        inp = Xf[:, 3:10] * wi[i]
        allt = np.concatenate([terms, inp[:, :, None]], 2).reshape(-1, 17)
        pre = P[:, 3:10, i].reshape(-1)
        med = np.median(np.abs(allt), 0); order = np.argsort(-med); a1, a2 = order[:2]
        mpre = float(np.median(np.abs(pre)))
        corr12 = float(np.corrcoef(allt[:, a1], allt[:, a2])[0, 1]) if allt[:, a1].std() > 0 and allt[:, a2].std() > 0 else np.nan
        sgn = np.sign(np.median(allt[:, a1])) * np.sign(np.median(allt[:, a2])) if True else 0
        opp = wh[i, a1] * wh[i, a2] < 0 if (a1 < 16 and a2 < 16) else False
        strict = bool(a1 in hid and a2 in hid and opp and corr12 < -0.7 and mpre < med[a1] / 3)
        # loose: any opposite-sign hiding pair, each term >= half median |pre|, sum's sd >= half pre's sd
        loose_pairs, best_pair = [], None
        for x in range(len(hid)):
            for y in range(x + 1, len(hid)):
                j, k = hid[x], hid[y]
                if wh[i, j] * wh[i, k] >= 0: continue
                s = allt[:, j] + allt[:, k]
                if min(med[j], med[k]) >= 0.5 * mpre and s.std() >= 0.5 * pre.std(): loose_pairs.append((j, k))
                if best_pair is None or min(med[j], med[k]) > best_pair[0]: best_pair = (min(med[j], med[k]), j, k)
        # gap fits for every hiding pair entering i
        for x in range(len(hid)):
            for y in range(x + 1, len(hid)):
                j, k = hid[x], hid[y]
                a = np.zeros(len(hid)); a[x], a[y] = wh[i, j], wh[i, k]
                r2w, nc = median_r2(cells, a, a @ Cov_all @ a)
                b = np.zeros(len(hid)); b[x], b[y] = 1 / sig[j], -1 / sig[k]
                r2u, _ = median_r2(cells, b, b @ Cov_all @ b)
                prs.append(dict(net=lab, memory=i, j=j, k=k, w_j=round(wh[i, j], 3), w_k=round(wh[i, k], 3),
                                opposite=bool(wh[i, j] * wh[i, k] < 0), med_term_j=round(med[j], 3), med_term_k=round(med[k], 3),
                                r2_weighted=round(r2w, 3), r2_weighted_gap_and_x=round(median_r2(cells, a, a @ Cov_all @ a, True)[0], 3), r2_unweighted_sd1=round(r2u, 3), cells=nc))
        col = wo[:, i]; mc = np.median(np.abs(col)); peak = float(np.abs(col).max() / mc) if mc > 0 else np.inf
        peakpos = [int(p) for p in np.where(np.abs(col) > 2 * mc)[0]]
        row = dict(net=lab, memory=i, top1=int(a1) if a1 < 16 else "input", top2=int(a2) if a2 < 16 else "input",
                   med_top1=round(med[a1], 3), med_top2=round(med[a2], 3), corr12=round(corr12, 3), med_pre=round(mpre, 3),
                   strict=strict, loose=bool(loose_pairs), loose_pairs=" ".join(f"{j}-{k}" for j, k in loose_pairs),
                   top_opp_pair=f"{best_pair[1]}-{best_pair[2]}" if best_pair else "", peak=round(peak, 2),
                   peak_positions=" ".join(map(str, peakpos)), readout=" ".join(f"{v:+.2f}" for v in col / mc) if mc > 0 else "")
        if best_pair:
            a = np.zeros(len(hid)); a[hid.index(best_pair[1])] = wh[i, best_pair[1]]; a[hid.index(best_pair[2])] = wh[i, best_pair[2]]
            row["top_pair_r2"] = round(median_r2(cells, a, a @ Cov_all @ a)[0], 3)
            row["top_pair_r2_gap_and_x"] = round(median_r2(cells, a, a @ Cov_all @ a, True)[0], 3)
        # substitutions (closed), costs in accuracy points on test lists
        def cost(js, kind):
            D = np.zeros((len(Xf), 10)); D[:, 1:] = H[:, :-1][:, :, js] @ wh[i, js]
            C = fit_sub(D, Vf, kind); return round(base - accuracy(run_sub(W, Xt, Vt, i, js, C, kind), Xt), 2)
        if hid:
            jbig = max(hid, key=lambda j: med[j]); row["largest_hiding_term"] = int(jbig)
            row["cost_flatten_largest_term"] = cost([jbig], "mean")
        if best_pair:
            js = [best_pair[1], best_pair[2]]
            row["cost_flatten_top_pair"] = cost(js, "mean")
            row["cost_gapformula_top_pair"] = cost(js, "gap")
        if strict and not (best_pair and {best_pair[1], best_pair[2]} == {int(a1), int(a2)}):
            row["cost_flatten_strict_pair"] = cost([int(a1), int(a2)], "mean")
        rows.append(row)
    nonflag = [r for r in rows if not r["strict"] and "largest_hiding_term" in r]
    pick = int(rng_pick.integers(len(nonflag))) if nonflag else None
    if pick is not None: nonflag[pick]["random_pick"] = True
    refs = DETAIL[lab].get(REFKEY, [])
    return dict(base=base, rows=rows, pairs=prs, refs=refs, hid=hid, mem=mem)


if __name__ == "__main__":
    nets = networks(); rngp = np.random.default_rng(5); res = {}
    only = sys.argv[1:]
    for lab, W in nets.items():
        if only and lab not in only: continue
        r = analyse(lab, W, rngp); res[lab] = r
        print(f"\n=== {lab}: accuracy {r['base']:.1f}, hiding {r['hid']}, memory {r['mem']}", flush=True)
        print("  hiding references (neuron, x largest, x second, const, R2): " + "; ".join(f"{f[0]}: {f[1]:.2f}, {f[2]:.2f}, {f[3]:.2f}, {f[4]:.2f}" for f in r["refs"]))
        for w in r["rows"]:
            print("  " + " ".join(f"{k}={v}" for k, v in w.items() if k not in ("net", "readout")), flush=True)
            print(f"     readout / median |readout|: {w['readout']}")
    tag = "_".join(only) if only else "all"
    keys = sorted({k for r in res.values() for w in r["rows"] for k in w}, key=lambda k: list(res[next(iter(res))]["rows"][0]).index(k) if k in res[next(iter(res))]["rows"][0] else 99)
    with open(os.path.join(HERE, f"sisters_{tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["base_accuracy"] + keys); wr.writeheader()
        for r in res.values():
            for w in r["rows"]: wr.writerow({"base_accuracy": round(r["base"], 2), **w})
    with open(os.path.join(HERE, f"pairs_{tag}.csv"), "w", newline="") as f:
        allp = [p for r in res.values() for p in r["pairs"]]
        if allp:
            wr = csv.DictWriter(f, fieldnames=list(allp[0])); wr.writeheader(); wr.writerows(allp)
    json.dump({lab: dict(base=r["base"], hid=r["hid"], mem=r["mem"], refs=r["refs"]) for lab, r in res.items()},
              open(os.path.join(HERE, f"nets_{tag}.json"), "w"), indent=1)
