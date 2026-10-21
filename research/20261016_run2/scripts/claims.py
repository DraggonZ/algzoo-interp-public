#!/usr/bin/env python
"""Re-run of the claims about ARC's AlgZoo model M16,10 that might be quoted publicly.

One section per claim, each with fixed seeds. Each section prints its result and replaces its entry in
reports/claims.json; `md` rebuilds reports/claims.md from the JSON.

Usage (from any directory):
    PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python claims.py C      # one section
    ... claims.py A B D                                                           # several
    ... claims.py all                                                             # A..N
    ... claims.py md                                                              # table only

Sections (claim numbers of the task brief):
    A base accuracies (1)              H noisy copies fail by position (8)
    B errors are fuzzy comparisons (2) I per-cell estimator (9)
    C errors made at the last input (3) J zero-background estimators Z_k (10)
    D negative 2nd max (4)             K comparison thresholds by age (11)
    E top-k variance curve (5)         L pointer fidelity (12)
    F sensitivity by weight block (6)  M comparator zeroing in the final state (13)
    G five sensitive weights (7)       N exact length-3 table (14)

Definitions are copied from the original scripts where they exist (named in each section); see
/data/output/research/algzoo/20261016_m16_10/scripts and .../reviews/review_1_checks.
Inputs are 10 iid N(0,1) values; "position" is 0..9; the target is the position of the 2nd largest.
Mid-sequence pointer readouts are used only from step 5 on (earlier readouts are unreliable).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import fcntl
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from scipy.stats import norm

from algzoo import exact_s2, exact_z2
from algzoo.bench import FWD_OPS, load_bench
from algzoo.core import accuracy, example, load, target, zoo

RUN = Path("/data/output/research/algzoo/20261016_run2")
JSON_PATH = RUN / "reports" / "claims.json"
MD_PATH = RUN / "reports" / "claims.md"
DEV = "cuda"
CLEAN = [1, 2, 4, 6, 7, 8]
DENSE = [j for j in range(16) if j not in CLEAN]
SIGMAS = ("mult0.005", "mult0.01", "mult0.02")
# the four degraded copies and one mild copy examined in review 1 (check6_noisy_anatomy.py: indices 5, 35, 40, 72, 80)
REVIEW_COPIES = ["mult0.005_5", "mult0.01_3", "mult0.01_8", "mult0.02_8", "mult0.02_16"]
DEGRADED4 = REVIEW_COPIES[1:]
th.backends.cuda.matmul.allow_tf32 = False


# ----------------------------------------------------------------------------------------------- helpers
def xstream(n, seed, chunk=2**21, dtype=th.float32):
    """n standard Gaussian inputs of length 10 in chunks, from one generator seeded once."""
    g = th.Generator(device=DEV).manual_seed(seed)
    done = 0
    while done < n:
        b = min(chunk, n - done)
        yield th.randn(b, 10, generator=g, device=DEV, dtype=dtype)
        done += b


def tparams(params, dtype=th.float32):
    return tuple(th.as_tensor(p, dtype=dtype, device=DEV) for p in params)


def run_states(tp, x):
    """Hidden states h_0..h_10 (list) for torch params tp = (w_hi, w_hh, w_oh)."""
    w_hi, w_hh, _ = tp
    h = th.zeros(x.shape[0], w_hh.shape[0], dtype=x.dtype, device=x.device)
    hs = [h]
    for t in range(x.shape[1]):
        h = th.relu(h @ w_hh.T + x[:, t, None] * w_hi)
        hs.append(h)
    return hs


def logits_of(tp, x):
    return run_states(tp, x)[-1] @ tp[2].T


def bse(p, n):
    return float(np.sqrt(p * (1 - p) / n))


def agree(new, old, se, unit):
    """Same value at the old value's precision: |new - old| <= 3 SE + half a unit of the old last digit."""
    return abs(new - old) <= 3 * se + unit / 2


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def bench_all():
    d = np.load(Path("/data/output/research/algzoo/20261016_m16_10/reports/benchmark.npz"))
    names = [str(n) for n in d["names"]]
    models = {n: (d["w_hi"][i], d["w_hh"][i], d["w_oh"][i]) for i, n in enumerate(names)}
    return names, dict(zip(names, d["acc"])), models


def group_of(name):
    return name.rsplit("_", 1)[0]


def rmse_table(est, acc, names, reps=2000, seed=0):
    """RMSE, bias, correlation against the true accuracies, overall and per noise level, with a
    bootstrap 95% interval over copies (resampled within each noise level)."""
    est, acc = np.asarray(est, float), np.asarray(acc, float)
    grp = np.array([group_of(n) for n in names])
    rng = np.random.default_rng(seed)
    out = {}
    for g in ("all",) + SIGMAS:
        idx = np.arange(len(est)) if g == "all" else np.flatnonzero(grp == g)
        e, a = est[idx], acc[idx]
        r = float(np.sqrt(np.mean((e - a) ** 2)))
        boots = []
        for _ in range(reps):
            if g == "all":
                bi = np.concatenate([rng.choice(np.flatnonzero(grp == s), (grp == s).sum()) for s in SIGMAS])
            else:
                bi = rng.choice(idx, len(idx))
            boots.append(np.sqrt(np.mean((est[bi] - acc[bi]) ** 2)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[g] = dict(rmse=r, rmse_ci=[float(lo), float(hi)], bias=float(np.mean(e - a)),
                      corr=float(np.corrcoef(e, a)[0, 1]), n=int(len(idx)),
                      samples_equiv=float(np.mean(a * (1 - a)) / r**2))
    return out


def row(rid, claim, value, n, se, status, old, how):
    return dict(id=rid, claim=claim, value=value, n=n, se=se, status=status, old=old, how=how)


def save(key, title, rows, details, seconds, notes=""):
    entry = dict(section=key, title=title, rows=rows, notes=notes, details=details,
                 seconds=round(seconds, 1), run_at=time.strftime("%Y-%m-%d %H:%M"))
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_PATH) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
        data[key] = _clean(entry)
        JSON_PATH.write_text(json.dumps(data, indent=1, default=_jsonable))
        fcntl.flock(lk, fcntl.LOCK_UN)
    for r in rows:
        print(f"  [{r['id']}] {r['status'].upper()}: {r['value']} ({r['se']}); old {r['old']}")
    if notes:
        print("  notes:", notes)


def _clean(o):
    """Convert numpy/torch values to plain JSON types; NaN and infinity become null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _jsonable(o):
    return str(o)


# ----------------------------------------------------------------------------------------------- A
def sec_A():
    """Claim 1: accuracies at 10^8 inputs (core.accuracy, float32)."""
    t0 = time.time()
    _, true_acc, _ = bench_all()
    n_s = 10**8
    p, n = accuracy(example(), n_s, seed=101)
    se = bse(p, n)
    old = 0.9531
    rows = [row("1a", "M16,10 (ARC's 432-parameter example model) puts its largest logit on the position of the "
                      "2nd-largest of 10 standard Gaussian inputs this often.",
                f"{p:.5f}", f"{n:,} random inputs", f"SE {se:.5f}",
                "reproduced" if agree(p, old, se, 1e-4) else "changed", "0.9531 (10^8 inputs)",
                f"Monte Carlo with core.accuracy, float32 on GPU, seed 101. An independent 2^29-input run "
                f"(benchmark.npz) gave {true_acc['example']:.5f}.")]
    olds = [0.910, 0.880, 0.949, 0.932, 0.890]
    zs, ses = [], []
    for s in range(5):
        a, n = accuracy(zoo(16, 10, s), n_s, seed=110 + s)
        zs.append(a)
        ses.append(bse(a, n))
    ok = all(agree(a, o, e, 1e-3) for a, o, e in zip(zs, olds, ses))
    rows.append(row("1b", "The five other trained 16-neuron, length-10 models in ARC's zoo (seeds 0-4) are less "
                          "accurate than the example model.",
                    ", ".join(f"{a:.4f}" for a in zs), f"{n_s:,} random inputs per model",
                    f"SE {max(ses):.5f} or less each", "reproduced" if ok else "changed",
                    ", ".join(f"{o:.3f}" for o in olds),
                    "core.accuracy on zoo(16, 10, seed), seeds 110-114. 2^29-input values: "
                    + ", ".join(f"{true_acc[f'zoo_seed{s}']:.4f}" for s in range(5)) + "."))
    det = dict(example=p, example_se=se, zoo=zs, zoo_se=ses, n=n_s,
               bench_2pow29=dict(example=true_acc["example"], zoo=[true_acc[f"zoo_seed{s}"] for s in range(5)]))
    save("A", "Base accuracies", rows, det, time.time() - t0)


# ----------------------------------------------------------------------------------------------- B
GAP_EDGES = [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]


def arrival_gap(x):
    """check3_gaps.gaps, verbatim: min over steps t >= 1 of |x_t - running max| and |x_t - running 2nd|."""
    n = x.shape[1]
    m1 = x[:, 0].clone()
    m2 = th.full_like(m1, -th.inf)
    g = th.full_like(m1, th.inf)
    for t in range(1, n):
        xt = x[:, t]
        g = th.minimum(g, (xt - m1).abs())
        g = th.minimum(g, th.where(th.isfinite(m2), (xt - m2).abs(), th.full_like(g, th.inf)))
        new_m1 = th.maximum(m1, xt)
        m2 = th.where(xt > m1, m1, th.maximum(m2, xt))
        m1 = new_m1
    return g


def gap_counts(params, n, seed, chunk):
    tp = tparams(params)
    inner = th.tensor(GAP_EDGES[1:-1], device=DEV)
    nb = len(GAP_EDGES) - 1
    cnt = th.zeros(nb, dtype=th.float64, device=DEV)
    cor = th.zeros(nb, dtype=th.float64, device=DEV)
    for x in xstream(n, seed, chunk):
        c = (logits_of(tp, x).argmax(-1) == target(x)).double()
        b = th.bucketize(arrival_gap(x), inner, right=True)
        cnt += th.bincount(b, minlength=nb).double()
        cor += th.bincount(b, weights=c, minlength=nb)
    return cnt.cpu().numpy(), cor.cpu().numpy()


def sec_B():
    """Claim 2: errors are fuzzy comparisons (definition of check3_gaps.py)."""
    t0 = time.time()
    p = example()
    # 1. the review's own run, exactly as check3_gaps.py drew it (2^22 inputs in one call, seed 5)
    cnt0, cor0 = gap_counts(p, 2**22, 5, 2**22)
    err0 = cnt0 - cor0
    share0_bins = err0 / err0.sum()
    review_printed = [round(float(s), 2) for s in share0_bins]
    review_sum_below01 = sum(review_printed[:3])
    review_exact_below01 = float(share0_bins[:3].sum())
    # 2. the main run
    n, seed = 2**26, 202
    cnt, cor = gap_counts(p, n, seed, 2**21)
    err = cnt - cor
    n_err = err.sum()
    s01 = err[:3].sum() / n_err
    acc = cor / cnt
    acc_se = [bse(a, c) for a, c in zip(acc, cnt)]
    a_lt001, a_01_02, a_ge05 = acc[0], acc[3], cor[6:].sum() / cnt[6:].sum()
    se_ge05 = bse(a_ge05, cnt[6:].sum())
    bins = [f"[{a},{b})" for a, b in zip(GAP_EDGES[:-1], GAP_EDGES[1:])]
    print("  bin, input share, accuracy, error share:")
    for i, b in enumerate(bins):
        print(f"   {b:12s} {cnt[i] / n:.4f} {acc[i]:.5f} {err[i] / n_err:.4f}")
    how_gap = ("Smallest arrival gap = min over steps t >= 1 of |x_t - running max| and |x_t - running 2nd max| "
               "(check3_gaps.py). Seed 202.")
    rows = [
        row("2a", "Most of M16,10's errors involve a close call: the share of errors in which some arriving input "
                  "came within 0.1 of the running max or running 2nd max.",
            pct(s01), f"{int(n_err):,} errors among {n:,} random inputs", f"SE {pct(bse(s01, n_err), 2)}",
            "reproduced" if agree(s01, 0.935, bse(s01, n_err), 0.001) else "changed",
            "93.5% (report); 94% (review 1)",
            how_gap),
        row("2b", "M16,10's accuracy on inputs whose smallest arrival gap is below 0.01 (a near-tie somewhere).",
            pct(a_lt001), f"{int(cnt[0]):,} inputs with gap < 0.01 (of {n:,})", f"SE {pct(acc_se[0], 2)}",
            "reproduced" if agree(a_lt001, 0.796, acc_se[0], 0.001) else "changed", "79.6%",
            "As 2a."),
        row("2c", "M16,10's accuracy on inputs whose smallest arrival gap is between 0.1 and 0.2.",
            pct(a_01_02, 2), f"{int(cnt[3]):,} inputs with 0.1 <= gap < 0.2", f"SE {pct(acc_se[3], 3)}",
            "reproduced" if agree(a_01_02, 0.990, acc_se[3], 0.001) else "changed", "99.0%",
            "As 2a."),
        row("2d", "M16,10's accuracy on inputs whose smallest arrival gap is at least 0.5 (no close call anywhere).",
            pct(a_ge05, 2), f"{int(cnt[6:].sum()):,} inputs with gap >= 0.5", f"SE {pct(se_ge05, 3)}",
            "reproduced" if agree(a_ge05, 0.999, se_ge05, 0.001) else "changed", "99.9% (report); 100% (review 1)",
            "As 2a."),
    ]
    det = dict(n=n, seed=seed, bins=bins, count=cnt, correct=cor, accuracy=acc, accuracy_se=acc_se,
               error_share=err / n_err, share_errors_gap_below_0_1=s01,
               review_rerun=dict(n=2**22, seed=5, error_share=share0_bins, printed=review_printed,
                                 sum_of_rounded=review_sum_below01, exact=review_exact_below01))
    save("B", "Errors are fuzzy comparisons", rows, det, time.time() - t0,
         f"Review 1's 94% is its three bin shares rounded first and then added "
         f"({' + '.join(f'{v:.2f}' for v in review_printed[:3])} = {review_sum_below01:.2f}); unrounded, its own "
         f"2^22-input run (seed 5) gives {pct(review_exact_below01)}. The two old figures therefore agree. "
         f"Review 1's '100% above 0.5' is the same rounding: accuracy is {pct(a_ge05, 2)}.")


# ----------------------------------------------------------------------------------------------- C
def sec_C():
    """Claim 3: errors made at the last input. No original script exists; definition below."""
    t0 = time.time()
    tp = tparams(example())
    w_oh = tp[2]
    n, seed = 2**24, 303
    k = dict(err=0, right9=0, last=0, c1=0, c2=0, c3=0, c4=0, c2_third=0, c1_close=0)
    start_counts = np.zeros(11)
    for x in xstream(n, seed, 2**20):
        hs = run_states(tp, x)
        right = {}
        for t in range(5, 11):
            ptr = (hs[t] @ w_oh.T).argmax(1)
            p2 = x[:, :t].argsort(1, descending=True)[:, 1]
            right[t] = ptr == p2 + (10 - t)
            if t == 10:
                ans = ptr
        err = ~right[10]
        # error start step: smallest t in 5..10 such that the pointer is wrong at every step t..10
        start = th.where(err, 10, -1)
        sw = err.clone()
        for t in range(9, 4, -1):
            sw = sw & ~right[t]
            start = th.where(sw, t, start)
        start_counts += th.bincount(start[err], minlength=11).cpu().numpy()
        o9 = x[:, :9].argsort(1, descending=True)
        v9 = x[:, :9].gather(1, o9)
        m1, m2 = v9[:, 0], v9[:, 1]
        x9 = x[:, 9]
        last = err & right[9]
        newmax = x9 > m1
        new2 = (x9 > m2) & ~newmax
        lower = x9 < m2
        p3_all = x.argsort(1, descending=True)[:, 2]
        c1 = last & lower & (ans == 9)
        c2 = last & lower & (ans != 9)
        k["err"] += int(err.sum())
        k["right9"] += int(right[9].sum())
        k["last"] += int(last.sum())
        k["c1"] += int(c1.sum())
        k["c2"] += int(c2.sum())
        k["c3"] += int((last & newmax).sum())
        k["c4"] += int((last & new2).sum())
        k["c2_third"] += int((c2 & (ans == p3_all)).sum())
        k["c1_close"] += int((c1 & (m2 - x9 < 0.1)).sum())
        del hs
    s_last = k["last"] / k["err"]
    split = [k[c] / k["last"] for c in ("c1", "c2", "c3", "c4")]
    split_se = [bse(s, k["last"]) for s in split]
    olds = [0.24, 0.38, 0.21, 0.17]
    ok = all(agree(s, o, e, 0.01) for s, o, e in zip(split, olds, split_se))
    how = ("Running answer after t inputs = argmax(W_oh h_t), right if on the running 2nd argmax s (padded slot "
           "s + 10 - t); last-input error = right after 9 inputs. Definition reconstructed (no script); seed 303.")
    starts = {("<=5" if t == 5 else str(t)): start_counts[t] / k["err"] for t in range(5, 11)}
    rows = [
        row("3a", "About half of M16,10's errors are made at the very last input: its running answer was right "
                  "after 9 inputs and the 10th input broke it.",
            pct(s_last), f"{k['err']:,} errors among {n:,} random inputs", f"SE {pct(bse(s_last, k['err']), 2)}",
            "reproduced" if agree(s_last, 0.51, bse(s_last, k["err"]), 0.01) else "changed", "51%", how),
        row("3b", "Errors made at the last input, split by what the last input was: below the running 2nd max and "
                  "taken for a new 2nd; below it, with the answer moving elsewhere (usually to the 3rd largest); "
                  "a new max whose old position is not moved into the answer; a new 2nd whose slot is not written.",
            " / ".join(pct(s, 0) for s in split), f"{k['last']:,} last-input errors",
            "SE " + " / ".join(pct(e, 1) for e in split_se), "reproduced" if ok else "changed", "24 / 38 / 21 / 17%",
            f"As 3a; groups by the last input's rank among the first nine and the final answer. In the second "
            f"group the answer is the 3rd largest in {pct(k['c2_third'] / max(k['c2'], 1), 0)} of cases."),
    ]
    det = dict(n=n, seed=seed, counts=k, share_last=s_last, split=split, split_se=split_se,
               pointer_right_after_9=k["right9"] / n, error_start_share=starts,
               c1_within_0_1_of_2nd=k["c1_close"] / max(k["c1"], 1),
               c2_answer_is_3rd=k["c2_third"] / max(k["c2"], 1))
    notes = ("Error start step (smallest t from 5 on after which the pointer stays wrong), share of errors: "
             + ", ".join(f"{t}: {pct(v)}" for t, v in starts.items())
             + ". Readouts before step 5 are unreliable, so earlier starts are lumped into '<=5'.")
    save("C", "Errors made at the last input", rows, det, time.time() - t0, notes)


# ----------------------------------------------------------------------------------------------- D
def sec_D():
    """Claim 4: inputs whose 2nd-largest value is negative."""
    t0 = time.time()
    tp = tparams(example())
    n, seed = 2**26, 404
    neg = negc = errs = errneg = 0
    for x in xstream(n, seed):
        c = logits_of(tp, x).argmax(-1) == target(x)
        m = x.topk(2, dim=-1).values[:, 1] < 0
        neg += int(m.sum())
        negc += int((c & m).sum())
        errs += int((~c).sum())
        errneg += int((~c & m).sum())
    mass_exact = 11 / 1024
    a = negc / neg
    rows = [
        row("4a", "Inputs whose 2nd-largest value is negative (at most one of the 10 is positive) are rare.",
            f"{pct(neg / n, 3)} sampled; exactly 11/1024 = {pct(mass_exact, 3)}", f"{n:,} random inputs",
            f"SE {pct(bse(neg / n, n), 3)}", "reproduced" if agree(mass_exact, 0.011, 0, 0.001) else "changed",
            "1.1%", "Exact: P(at most one of 10 positive) = 11/2^10; sampled check with seed 404."),
        row("4b", "M16,10 is mostly wrong on inputs whose 2nd-largest value is negative.",
            pct(a), f"{neg:,} such inputs among {n:,}", f"SE {pct(bse(a, neg), 2)}",
            "reproduced" if agree(a, 0.22, bse(a, neg), 0.01) else "changed", "22%",
            f"Monte Carlo, seed 404. These inputs hold {pct(errneg / errs)} of all errors."),
    ]
    det = dict(n=n, seed=seed, n_neg=neg, acc_neg=a, mass_sampled=neg / n, mass_exact=mass_exact,
               error_share=errneg / errs)
    save("D", "Negative 2nd max", rows, det, time.time() - t0)


# ----------------------------------------------------------------------------------------------- E
def sec_E():
    """Claim 5: share of the variance of correctness explained by fixing the top-k values and positions."""
    t0 = time.time()
    tp = tparams(example())
    A, R, seed, batch = 2**16, 256, 505, 4096
    olds = [0.040, 0.250, 0.621, 0.757, 0.829, 0.873, 0.903, 0.933, 0.963]
    g = th.Generator(device=DEV).manual_seed(seed)
    res = []
    for k in range(1, 10):
        phat = []
        for _ in range(A // batch):
            x0 = th.randn(batch, 10, generator=g, device=DEV, dtype=th.float64)
            order = x0.argsort(1, descending=True)
            top = th.zeros_like(x0, dtype=th.bool).scatter_(1, order[:, :k], True)
            mk = x0.gather(1, order[:, k - 1:k])                          # k-th largest value, [B, 1]
            u = th.rand(batch, R, 10, generator=g, device=DEV, dtype=th.float64)
            low = th.special.ndtri((u * th.special.ndtr(mk)[:, :, None]).clamp(min=1e-300))
            X = th.where(top[:, None, :], x0[:, None, :], low).reshape(-1, 10).float()
            c = (logits_of(tp, X).argmax(-1) == target(X)).reshape(batch, R).double()
            phat.append(c.mean(1))
        ph = th.cat(phat).cpu().numpy()
        res.append(ph)
    rng = np.random.default_rng(0)
    out = []
    for k, ph in zip(range(1, 10), res):
        def share(v):
            P = v.mean()
            between = v.var(ddof=1) - np.mean(v * (1 - v)) / (R - 1)
            return between / (P * (1 - P)), v.var(ddof=0) / (P * (1 - P))
        s, naive = share(ph)
        boots = [share(ph[rng.integers(0, A, A)])[0] for _ in range(300)]
        out.append(dict(k=k, share=s, se=float(np.std(boots)), naive=naive, acc=float(ph.mean())))
        print(f"  k={k}: share {s:.4f} ± {np.std(boots):.4f} (naive {naive:.4f})")
    vals = [o["share"] for o in out]
    ses = [o["se"] for o in out]
    ok = all(agree(v, o, e, 0.01) for v, o, e in zip(vals, olds, ses))
    rows = [row("5", "Every input matters to M16,10: share of the variance of correctness explained by fixing the "
                     "values and positions of the k largest inputs, for k = 1..9 (the rest resampled below them).",
                ", ".join(f"{v:.3f}" for v in vals), f"{A:,} random anchor inputs x {R} resamples each, per k",
                "SE " + ", ".join(f"{e:.3f}" for e in ses), "reproduced" if ok else "changed",
                ", ".join(f"{o:.3f}" for o in olds),
                "Between-anchor variance of the resampled accuracy, minus its sampling noise mean(p(1-p))/(R-1), "
                "over p(1-p). SE by bootstrap over anchors. Seed 505.")]
    save("E", "Top-k variance curve", rows, dict(A=A, R=R, seed=seed, per_k=out), time.time() - t0,
         "Without the noise correction the shares are higher by about (1 - share)/256: "
         + ", ".join(f"{o['naive']:.3f}" for o in out) + ".")


# ----------------------------------------------------------------------------------------------- F
def acc_crn(params, n, seed, chunk=2**21):
    """Accuracy on a fixed input stream (common random numbers across models)."""
    tp = tparams(params)
    hits = 0
    for x in xstream(n, seed, chunk):
        hits += int((logits_of(tp, x).argmax(-1) == target(x)).sum())
    return hits / n


def block_masks():
    """Weight blocks: (index into (w_hi, w_hh, w_oh), boolean mask). Rows receive from columns in W_hh."""
    w_hi0, w_hh0, w_oh0 = example()
    out = {}

    def mk(w, rows=None, cols=None):
        m = np.zeros(w.shape, dtype=bool)
        if w.ndim == 1:
            m[rows] = True
        else:
            m[np.ix_(rows, cols)] = True
        return m
    out["W_hh among the six clean neurons (36 weights)"] = (1, mk(w_hh0, CLEAN, CLEAN))
    out["W_hh into clean neurons from the 10 others (60)"] = (1, mk(w_hh0, CLEAN, DENSE))
    out["W_hh into the 10 others from clean neurons (60)"] = (1, mk(w_hh0, DENSE, CLEAN))
    out["W_hh among the 10 other neurons (100)"] = (1, mk(w_hh0, DENSE, DENSE))
    out["input weights of the clean neurons (6)"] = (0, mk(w_hi0, CLEAN))
    out["input weights of the 10 others (10)"] = (0, mk(w_hi0, DENSE))
    out["readout weights from clean neurons (60)"] = (2, mk(w_oh0, list(range(10)), CLEAN))
    out["readout weights from the 10 others (100)"] = (2, mk(w_oh0, list(range(10)), DENSE))
    return out


def sec_F():
    """Claim 6: 1% multiplicative noise on one block of weights at a time, 48 draws per block (4 sets of 12)."""
    t0 = time.time()
    base = example()
    n, xseed, n_draw = 2**24, 606, 48
    a0 = acc_crn(base, n, xseed)
    res = {}
    for bi, (bname, (wi, mask)) in enumerate(block_masks().items()):
        rng = np.random.default_rng(6060 + bi)
        accs = []
        for _ in range(n_draw):
            ws = [w.copy() for w in base]
            noise = 1 + 0.01 * rng.standard_normal(ws[wi].shape)
            ws[wi] = np.where(mask, ws[wi] * noise, ws[wi])
            accs.append(acc_crn(tuple(ws), n, xseed))
        accs = np.array(accs)
        sets = accs.reshape(4, 12)
        res[bname] = dict(acc=accs, median=np.median(accs), p10=np.percentile(accs, 10),
                          p90=np.percentile(accs, 90), min=accs.min(), max=accs.max(),
                          set_ranges=[[s.min(), s.max()] for s in sets], max_drop=a0 - accs.min(),
                          median_drop=a0 - np.median(accs))
        print(f"  {bname:50s} median {np.median(accs):.3f}, 10-90% {res[bname]['p10']:.3f}-{res[bname]['p90']:.3f},"
              f" range {accs.min():.3f}-{accs.max():.3f}; 12-draw ranges "
              + " ".join(f"{a:.2f}-{b:.2f}" for a, b in res[bname]["set_ranges"]))
    cname = "W_hh among the six clean neurons (36 weights)"
    c = res[cname]
    others = {k: v for k, v in res.items() if k != cname}
    se1 = bse(a0, n)
    r1 = c["set_ranges"][0]
    ok1 = abs(r1[0] - 0.73) <= 0.03 and abs(r1[1] - 0.90) <= 0.03
    touch = {k: v for k, v in others.items() if "clean" in k and "into clean" not in k}
    apart = {k: v for k, v in others.items() if k not in touch}
    big = {k: v for k, v in others.items() if v["max_drop"] > 0.035}
    rows = [
        row("6a", "M16,10 is fragile in its six clean comparator neurons: multiplying each of the 36 recurrent "
                  "weights among neurons 1, 2, 4, 6, 7, 8 by 1 + 0.01 N(0,1) drops accuracy from 0.953 to this "
                  "spread over random draws.",
            f"median {c['median']:.3f}, 10th-90th percentile {c['p10']:.3f}-{c['p90']:.3f}; min-max of four sets of "
            "12 draws: " + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in c["set_ranges"]),
            f"{n_draw} noise draws; {n:,} common inputs per draw",
            f"Monte Carlo SE per draw {se1:.4f}; spread is between draws",
            "reproduced" if ok1 else "changed", "0.73-0.90 (min-max of 12 draws)",
            "Weights x (1 + 0.01 N(0,1)); same inputs for every draw. Draw seed 6060, input seed 606."),
        row("6b", "The same 1% noise on M16,10's other weights costs less, but the clean neurons' input weights and "
                  "outgoing weights still matter: the largest accuracy drop in any of 48 draws, per block of weights.",
            "clean neurons' input and outgoing weights: " + "; ".join(f"{k} {100 * v['max_drop']:.1f} points" for k, v in touch.items())
            + ". All other blocks: " + "; ".join(f"{k} {100 * v['max_drop']:.1f}" for k, v in apart.items()),
            f"{n_draw} draws per block; {n:,} common inputs per draw", f"Monte Carlo SE per draw {se1:.4f}",
            "reproduced" if not big else "changed", "at most 2-3 points (blocks not recorded)",
            "As 6a, one block at a time; block split is this script's choice (the original was not recorded). "
            "Draw seeds 6061-6067."),
    ]
    det = dict(n=n, base_acc=a0, blocks=res)
    save("F", "Sensitivity by weight block", rows, det, time.time() - t0,
         "The old range 0.73-0.90 is one set of 12 draws; four sets of 12 give the ranges in row 6a, so quote the "
         "median and percentiles instead. Median drops per other block: "
         + "; ".join(f"{k} {100 * v['median_drop']:.1f}" for k, v in others.items())
         + " points. Blocks that do not involve the clean neurons' input or outgoing weights never cost more than "
           f"{100 * max(v['max_drop'] for v in apart.values()):.1f} points.")


# ----------------------------------------------------------------------------------------------- G
def sec_G():
    """Claim 7: single +-1% changes to the 36 W_hh weights among the clean neurons (check7_sensitivity.py),
    on common inputs, with the SE of each paired difference."""
    t0 = time.time()
    base = example()
    n, seed, chunk = 2**24, 707, 2**21
    xs_c0 = []
    tp0 = tparams(base)
    for x in xstream(n, seed, chunk):
        xs_c0.append((logits_of(tp0, x).argmax(-1) == target(x)))
    a0 = sum(int(c.sum()) for c in xs_c0) / n

    def paired(params):
        tp = tparams(params)
        gain = loss = 0
        for x, c0 in zip(xstream(n, seed, chunk), xs_c0):
            c = logits_of(tp, x).argmax(-1) == target(x)
            gain += int((c & ~c0).sum())
            loss += int((~c & c0).sum())
        d = (gain - loss) / n
        se = np.sqrt(((gain + loss) / n - d**2) / n)
        return d, se

    rows_ = []
    for i in CLEAN:
        for j in CLEAN:
            for sgn in (+1, -1):
                w_hi, w_hh, w_oh = (w.copy() for w in base)
                w_hh[i, j] *= 1 + sgn * 0.01
                d, se = paired((w_hi, w_hh, w_oh))
                rows_.append(dict(i=i, j=j, sign=sgn, diff=d, se=se))
    rows_.sort(key=lambda r: r["diff"])
    big = [r for r in rows_ if r["diff"] < -0.01]
    borderline = [r for r in rows_ if abs(r["diff"] + 0.01) < 3 * r["se"]]
    weights = sorted({(r["i"], r["j"]) for r in big})
    both = sorted({w for w in weights if sum((r["i"], r["j"]) == w for r in big) == 2})
    for r in rows_[:12]:
        print(f"  W_hh[{r['i']},{r['j']}] {'+' if r['sign'] > 0 else '-'}1%: {100 * r['diff']:+.3f} ± {100 * r['se']:.3f} points")
    old_w = [(7, 2), (2, 2), (8, 7), (1, 7), (8, 2)]
    old_both = [(7, 2), (2, 2), (8, 7), (1, 7)]
    ok = len(big) == 9 and set(weights) == set(old_w) and set(both) == set(old_both)
    wtxt = ", ".join(f"[{i},{j}]" for i, j in weights)
    # n2's self-weight x 0.98 by age of the max (review 1, finding 3)
    tp = tparams((base[0], base[1] * np.where(np.eye(16, dtype=bool) & (np.arange(16)[:, None] == 2), 0.98, 1.0),
                  base[2]))
    hit_age = np.zeros(10)
    cnt_age = np.zeros(10)
    for x in xstream(n, seed, chunk):
        c = (logits_of(tp, x).argmax(-1) == target(x)).double()
        age1 = 9 - x.argmax(-1)
        hit_age += th.bincount(age1, weights=c, minlength=10).cpu().numpy()
        cnt_age += th.bincount(age1, minlength=10).cpu().numpy()
    acc_age = hit_age / cnt_age
    rows = [
        row("7a", "A handful of M16,10's weights carry its sensitivity: the number of the 72 single +-1% changes to "
                  "recurrent weights among the six clean neurons that cost more than 1 accuracy point, and the "
                  "largest cost.",
            f"{len(big)} of 72; largest {100 * -rows_[0]['diff']:.2f} points",
            f"72 perturbed models x {n:,} common inputs",
            f"paired SE {100 * max(r['se'] for r in big):.3f} points or less; {len(borderline)} changes within 3 SE "
            f"of the 1-point line",
            "reproduced" if (len(big) == 9 and abs(-rows_[0]["diff"] - 0.025) < 0.0025) else "changed",
            "9 of 72; up to 2.5 points",
            "check7_sensitivity.py with 8x more inputs (seed 707). Paired SE = sd(c_new - c_base)/sqrt(n)."),
        row("7b", "The costly single changes all involve the same few recurrent weights (row receives from column), "
                  "and for some of them both +1% and -1% cost more than 1 point (a sharp optimum).",
            f"weights {wtxt}; both directions: " + ", ".join(f"[{i},{j}]" for i, j in both),
            "the changes above", "see 7a", "reproduced" if ok else "changed",
            "[7,2], [2,2], [8,7], [1,7], [8,2]; both directions for the first four", "As 7a."),
    ]
    det = dict(n=n, seed=seed, base_acc=a0, changes=rows_, n2_self_x098_acc_by_age_of_max=acc_age,
               n2_self_x098_n_by_age=cnt_age)
    save("G", "Five sensitive weights", rows, det, time.time() - t0,
         "Review 1 also reported n2's self-weight x 0.98 giving accuracy 0.80 when the max is 9 steps old and 0.93 "
         f"when it just arrived; this run gives {acc_age[9]:.3f} and {acc_age[0]:.3f}.")


# ----------------------------------------------------------------------------------------------- H
def share_corrected(cnt, cor):
    """Bias-corrected between-cell share of Var(c): sum_c (n_c/N)(m_c - p)^2 minus its expected sampling
    noise sum_c s_c^2 (1 - n_c/N)/N, divided by p(1-p). Also returns the naive share."""
    N = cnt.sum()
    p = cor.sum() / N
    m = np.where(cnt > 0, cor / np.maximum(cnt, 1), 0)
    naive = np.sum(cnt * (m - p) ** 2) / N
    s2 = np.where(cnt > 1, m * (1 - m) * cnt / np.maximum(cnt - 1, 1), 0)
    corr = naive - np.sum(s2 * (1 - cnt / N)) / N
    return corr / (p * (1 - p)), naive / (p * (1 - p))


def sec_H():
    """Claim 8: noisy copies fail by position (check6_noisy_anatomy.py), with a bias-corrected share."""
    t0 = time.time()
    names, true_acc, models = bench_all()
    sel = ["example"] + [n for n in names if n.startswith("mult")]
    tps = [tparams(models[m]) for m in sel]
    n, seed, chunk = 2**23, 808, 2**19
    nb = n // chunk
    K3 = np.zeros((len(sel), nb, 1000))
    C3 = np.zeros((len(sel), nb, 1000))
    K2 = np.zeros((len(sel), nb, 100))
    C2 = np.zeros((len(sel), nb, 100))
    age2_h = np.zeros((len(sel), 10))
    age1_h = np.zeros((len(sel), 10))
    age_n2 = np.zeros(10)
    age_n1 = np.zeros(10)
    for b, x in enumerate(xstream(n, seed, chunk)):
        order = x.argsort(-1, descending=True)
        y = order[:, 1]
        k3 = order[:, 0] * 100 + order[:, 1] * 10 + order[:, 2]
        k2 = order[:, 0] * 10 + order[:, 1]
        a2, a1 = 9 - order[:, 1], 9 - order[:, 0]
        age_n2 += th.bincount(a2, minlength=10).cpu().numpy()
        age_n1 += th.bincount(a1, minlength=10).cpu().numpy()
        cnt3 = th.bincount(k3, minlength=1000).cpu().numpy()
        cnt2 = th.bincount(k2, minlength=100).cpu().numpy()
        for mi, tp in enumerate(tps):
            c = (logits_of(tp, x).argmax(-1) == y).double()
            K3[mi, b] = cnt3
            C3[mi, b] = th.bincount(k3, weights=c, minlength=1000).cpu().numpy()
            K2[mi, b] = cnt2
            C2[mi, b] = th.bincount(k2, weights=c, minlength=100).cpu().numpy()
            age2_h[mi] += th.bincount(a2, weights=c, minlength=10).cpu().numpy()
            age1_h[mi] += th.bincount(a1, weights=c, minlength=10).cpu().numpy()
    res = {}
    for mi, m in enumerate(sel):
        s3, nv3 = share_corrected(K3[mi].sum(0), C3[mi].sum(0))
        s2, _ = share_corrected(K2[mi].sum(0), C2[mi].sum(0))
        bs3 = [share_corrected(K3[mi, b], C3[mi, b])[0] for b in range(nb)]
        res[m] = dict(acc=C3[mi].sum() / n, true_acc=true_acc[m], share_p123=s3, share_p123_naive=nv3,
                      share_p123_se=float(np.std(bs3, ddof=1) / np.sqrt(nb)), share_p12=s2,
                      acc_by_age2=age2_h[mi] / age_n2, acc_by_age1=age1_h[mi] / age_n1)
    for m in ["example"] + REVIEW_COPIES:
        r = res[m]
        print(f"  {m:14s} acc {r['acc']:.3f}  share (p1,p2,p3) {r['share_p123']:.4f} ± {r['share_p123_se']:.4f}"
              f" (naive {r['share_p123_naive']:.3f}), (p1,p2) {r['share_p12']:.3f}")
    deg = [res[m]["share_p123"] for m in DEGRADED4]
    top3 = sorted(deg)[1:]
    a6 = res["mult0.02_8"]["acc_by_age2"][6]
    a6_se = bse(a6, age_n2[6])
    noisy = [m for m in sel if m != "example"]
    allsh = np.array([res[m]["share_p123"] for m in noisy])
    acc_noisy = np.array([res[m]["true_acc"] for m in noisy])
    ok = (abs(min(top3) - 0.39) <= 0.03 and abs(max(top3) - 0.61) <= 0.03 and abs(min(deg) - 0.10) <= 0.03
          and abs(res["example"]["share_p123"] - 0.03) <= 0.015)
    rows = [
        row("8a", "Weight-noise copies of M16,10 fail by position: the share of the variance of correctness "
                  "explained by the positions of the three largest inputs, in four degraded copies and in the "
                  "base model.",
            ", ".join(f"{m} {res[m]['share_p123']:.2f}" for m in DEGRADED4)
            + f"; base {res['example']['share_p123']:.3f}",
            f"{n:,} common random inputs per model, 1,000 position cells",
            "SE " + ", ".join(f"{res[m]['share_p123_se']:.4f}" for m in DEGRADED4)
            + f"; base {res['example']['share_p123_se']:.4f}",
            "reproduced" if ok else "changed",
            "39-61% in three of four degraded copies, 10% in the fourth, 3% in the base model",
            "check6_noisy_anatomy.py's between-cell share, minus the sampling noise of the cell means "
            "(bias-corrected); SE from 16 independent batches. Copies are the four review 1 examined. Seed 808."),
        row("8b", "In the noisy copy mult0.02_8 (true accuracy 0.614), accuracy collapses when the 2nd largest "
                  "input is 6 steps old (at position 3).",
            f"{a6:.3f}", f"{int(age_n2[6]):,} inputs with the 2nd max at position 3", f"SE {a6_se:.4f}",
            "reproduced" if agree(a6, 0.16, a6_se, 0.01) else "changed", "0.16",
            "Accuracy by age of the 2nd max (age = 9 - position), check6_noisy_anatomy.py. Seed 808."),
    ]
    det = dict(n=n, seed=seed, per_model=res,
               per_sigma_median_share={s: float(np.median([res[m]["share_p123"] for m in noisy if group_of(m) == s]))
                                       for s in SIGMAS})
    save("H", "Noisy copies fail by position", rows, det, time.time() - t0,
         f"Across all 96 noisy copies the share explained by the top-3 positions has median {np.median(allsh):.2f} "
         f"and range {allsh.min():.2f}-{allsh.max():.2f} (SE per copy at most {max(res[m]['share_p123_se'] for m in noisy):.4f}); "
         f"its correlation with true accuracy is {np.corrcoef(allsh, acc_noisy)[0, 1]:.2f}. "
         "Naive (uncorrected) shares in the five review copies: "
         + ", ".join(f"{m} {res[m]['share_p123_naive']:.3f}" for m in REVIEW_COPIES)
         + f"; base {res['example']['share_p123_naive']:.3f}; the noise correction is below 0.001 at 2^23 inputs. "
           f"The mild copy mult0.005_5 (true accuracy 0.932) gives {res['mult0.005_5']['share_p123']:.3f}.")


# ----------------------------------------------------------------------------------------------- I
def order_stat_moments(seed, draws=10**7, n=10):
    rng = np.random.default_rng(seed)
    mu = np.zeros(n)
    m2 = np.zeros((n, n))
    done = 0
    while done < draws:
        b = min(10**6, draws - done)
        S = -np.sort(-rng.standard_normal((b, n)), axis=1)
        mu += S.sum(0)
        m2 += S.T @ S
        done += b
    return mu / draws, m2 / draws


def cell_moments(k, MU, M2, n=10):
    """pilot_cells.cell_moments, verbatim apart from passing MU and M2."""
    low = list(range(k, n))
    mean = np.concatenate([MU[:k], np.full(n - k, MU[low].mean())])
    E = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < k and j < k:
                E[i, j] = M2[i, j]
            elif i < k:
                E[i, j] = M2[i, low].mean()
            elif j < k:
                E[i, j] = M2[low, j].mean()
            elif i == j:
                E[i, j] = np.mean([M2[a, a] for a in low])
            else:
                E[i, j] = np.mean([M2[a, b] for a in low for b in low if a != b])
    return mean, E - np.outer(mean, mean)


def cell_estimate(params, k, MU, M2, n=10):
    """pilot_cells.estimate, verbatim apart from passing MU and M2: mean over cells of Phi(margin / width)."""
    w_hi, w_hh, w_oh = (th.tensor(p) for p in params)
    mean_c, cov_c = cell_moments(k, MU, M2)
    cells = [c for c in itertools.permutations(range(n), k)]
    X = th.zeros(len(cells), n, dtype=th.float64)
    perm = np.zeros((len(cells), n), dtype=int)
    for ci, c in enumerate(cells):
        rest = [p for p in range(n) if p not in c]
        order = list(c) + rest
        for slot, pos in enumerate(order):
            perm[ci, pos] = slot
        X[ci] = th.tensor(mean_c[perm[ci]])
    X.requires_grad_(True)
    h = th.zeros(len(cells), w_hh.shape[0], dtype=th.float64)
    for t in range(n):
        h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
    logits = h @ w_oh.T
    tgt = th.tensor([c[1] for c in cells])
    lt = logits.gather(1, tgt[:, None])[:, 0]
    other = logits.scatter(1, tgt[:, None], -th.inf).max(1).values
    mu = lt - other
    g, = th.autograd.grad(mu.sum(), X)
    mu = mu.detach().numpy()
    g = g.numpy()
    s = np.array([np.sqrt(g[ci] @ cov_c[np.ix_(perm[ci], perm[ci])] @ g[ci]) for ci in range(len(cells))])
    ops = len(cells) * 3 * FWD_OPS
    return norm.cdf(mu / np.maximum(s, 1e-12)).mean(), (mu > 0).mean(), ops


def sec_I():
    """Claim 9: per-cell estimator of review 1 (pilot_cells.py, top-3 cells, probit variant)."""
    t0 = time.time()
    names, acc, models = load_bench(("mult",))
    names = [str(x) for x in names]
    MU, M2 = order_stat_moments(0)
    est = np.array([cell_estimate(m, 3, MU, M2)[0] for m in models])
    ops = cell_estimate(models[0], 3, MU, M2)[2]
    base = cell_estimate(example(), 3, MU, M2)[0]
    tab = rmse_table(est, acc, names)
    MU1, M21 = order_stat_moments(1)
    est1 = np.array([cell_estimate(m, 3, MU1, M21)[0] for m in models])
    d_seed = float(np.sqrt(np.mean((est1 - acc) ** 2)) - tab["all"]["rmse"])
    for g, v in tab.items():
        print(f"  {g:10s} RMSE {v['rmse']:.4f} [{v['rmse_ci'][0]:.4f}, {v['rmse_ci'][1]:.4f}] bias {v['bias']:+.4f}"
              f" corr {v['corr']:.3f}")
    a = tab["all"]
    ok = (agree(a["rmse"], 0.066, 0, 0.001) and agree(a["bias"], 0.054, 0, 0.001)
          and all(agree(tab[s]["bias"], o, 0, 0.001) for s, o in zip(SIGMAS, (0.011, 0.055, 0.095))))
    rows = [
        row("9a", "Review 1's per-cell estimator (each of the 720 top-3 position cells gets its mean input; "
                  "accuracy = mean of Phi(margin/width)) misses the true accuracies of the 96 noisy copies by this "
                  "root-mean-square error, overestimating on average.",
            f"RMSE {a['rmse']:.3f}, bias {a['bias']:+.3f}, correlation {a['corr']:.2f}",
            "96 noisy copies (32 per noise level)",
            f"RMSE 95% bootstrap interval over copies {a['rmse_ci'][0]:.3f}-{a['rmse_ci'][1]:.3f}",
            "reproduced" if ok else "changed", "RMSE 0.066, bias +0.054, correlation 0.98",
            f"pilot_cells.py (k = 3, probit), order-statistic moments from 10^7 draws with seed 0; with seed 1 the "
            f"RMSE moves by {d_seed:+.5f}. True accuracies from 2^29 inputs."),
        row("9b", "The per-cell estimator's error by noise level (weights multiplied by 1 + sigma N(0,1)).",
            "; ".join(f"sigma {s[4:]}: RMSE {tab[s]['rmse']:.3f}, bias {tab[s]['bias']:+.3f}" for s in SIGMAS),
            "32 noisy copies per noise level",
            "RMSE intervals " + "; ".join(f"{tab[s]['rmse_ci'][0]:.3f}-{tab[s]['rmse_ci'][1]:.3f}" for s in SIGMAS),
            "reproduced" if ok else "changed", "bias +0.011, +0.055, +0.095 (RMSE per sigma not reported)",
            "As 9a."),
        row("9c", "The per-cell estimator is worth only a few dozen samples: the number of random samples whose "
                  "expected RMSE equals its RMSE, against its cost in forward passes.",
            f"{a['samples_equiv']:.0f} samples; cost {ops / FWD_OPS:,.0f} forward passes, at which sampling would have "
            f"RMSE {np.sqrt(np.mean(acc * (1 - acc)) / (ops / FWD_OPS)):.4f}",
            "96 noisy copies", "-", "reproduced" if abs(a["samples_equiv"] - 35) <= 5 else "changed",
            "about 35 samples, 2,160 forward passes",
            "Samples-equivalent = mean(p(1-p))/RMSE^2 over the 96 copies. Cost = 720 cells x 3 passes."),
    ]
    det = dict(table=tab, base_estimate=base, ops=ops, rmse_seed1=float(np.sqrt(np.mean((est1 - acc) ** 2))),
               estimates=dict(zip(names, est)))
    save("I", "Per-cell estimator", rows, det, time.time() - t0,
         f"On the base model the estimator gives {base:.3f} (true 0.953).")


# ----------------------------------------------------------------------------------------------- J
@th.no_grad()
def z_sampled(params, k, n, seed, chunk=2**21):
    tp = tparams(params)
    hits = 0
    for x in xstream(n, seed, chunk):
        v, idx = x.topk(k, dim=-1)
        xk = th.zeros_like(x).scatter_(1, idx, v)
        hits += int((logits_of(tp, xk).argmax(-1) == idx[:, 1]).sum())
    return hits / n


def sec_J():
    """Claim 10: zero-background estimators Z_k (all but the top k inputs set to zero; target = original
    2nd position). Z_2 exact (exact_z2.py), Z_3 and Z_4 sampled with 2^25 inputs per copy."""
    t0 = time.time()
    names, acc, models = load_bench(("mult",))
    names = [str(x) for x in names]
    n = 2**25
    z2, cells = [], []
    for m in models:
        z, _, nc = exact_z2.exact_z2(m)
        z2.append(z)
        cells.append(nc)
    z3 = [z_sampled(m, 3, n, 3000 + i) for i, m in enumerate(models)]
    z4 = [z_sampled(m, 4, n, 4000 + i) for i, m in enumerate(models)]
    base = dict(z2=exact_z2.exact_z2(example())[0], z3=z_sampled(example(), 3, n, 3999),
                z4=z_sampled(example(), 4, n, 4999))
    tabs = {k: rmse_table(z, acc, names) for k, z in (("Z2", z2), ("Z3", z3), ("Z4", z4))}
    for k, t in tabs.items():
        print(f"  {k}: " + "; ".join(f"{g} RMSE {v['rmse']:.4f} bias {v['bias']:+.4f}" for g, v in t.items()))
    olds = dict(Z2=0.122, Z3=0.055, Z4=0.050)
    rows = []
    for rid, k in (("10a", "Z2"), ("10b", "Z3"), ("10c", "Z4")):
        t = tabs[k]
        kk = k[1]
        rows.append(row(
            rid, f"Keeping only the {kk} largest inputs (all others set to zero, target still the original 2nd "
                 f"position) gives an accuracy Z_{kk} that misses the true accuracies of the 96 noisy copies by "
                 f"this RMSE, overestimating them.",
            f"RMSE {t['all']['rmse']:.3f} (bias {t['all']['bias']:+.3f}); by sigma 0.005/0.01/0.02: "
            + " / ".join(f"{t[s]['rmse']:.3f}" for s in SIGMAS),
            "96 noisy copies (32 per noise level)",
            f"95% bootstrap interval over copies {t['all']['rmse_ci'][0]:.3f}-{t['all']['rmse_ci'][1]:.3f}",
            "reproduced" if agree(t["all"]["rmse"], olds[k], 0, 0.001) else "changed", f"{olds[k]:.3f}",
            ("Exact, from the weights: exact_z2.py (angle intervals, first-index tie rule)." if k == "Z2" else
             f"Sampled, {n:,} inputs per copy (Monte Carlo SE of each Z_{kk} under 1e-4), seeds "
             f"{3000 if k == 'Z3' else 4000}+copy index.") + f" Base model Z_{kk} = {base[k.lower()]:.4f}."))
    det = dict(n=n, z2=dict(zip(names, z2)), z3=dict(zip(names, z3)), z4=dict(zip(names, z4)), cells=cells,
               base=base, tables=tabs)
    save("J", "Zero-background estimators", rows, det, time.time() - t0)


# ----------------------------------------------------------------------------------------------- K
def verdicts(params, n, seed, t_min=5, chunk=2**20):
    """thresholds.verdicts, chunked (same definition)."""
    tp = tparams(params, th.float64)
    w_oh = tp[2]
    rows = {"max": [], "2nd": []}
    for x in xstream(n, seed, chunk, th.float64):
        H = run_states(tp, x)
        for t in range(t_min, 11):
            pre = x[:, : t - 1]
            o = pre.argsort(1, descending=True)
            v = pre.gather(1, o)
            m1, m2 = v[:, 0], v[:, 1]
            p1, p2 = o[:, 0], o[:, 1]
            xn = x[:, t - 1]
            ptr = (H[t] @ w_oh.T).argmax(1)
            age1 = (t - 2) - p1
            age2 = (t - 2) - p2
            sel = (xn > m2) & (m1 > 0) & ((xn / m1 - 1).abs() < 0.15)
            newmax_v = ptr == p1 + (10 - t)
            new2_v = ptr == 9
            keep = sel & (newmax_v | new2_v)
            rows["max"].append(th.stack([(xn / m1)[keep], newmax_v[keep].double(), age1[keep].double()], 1).cpu())
            sel = (xn < m1) & (m2 > 0) & ((xn / m2 - 1).abs() < 0.15)
            new2 = ptr == 9
            lower = ptr == p2 + (10 - t)
            keep = sel & (new2 | lower)
            rows["2nd"].append(th.stack([(xn / m2)[keep], new2[keep].double(), age2[keep].double()], 1).cpu())
        del H
    return {k: th.cat(v).numpy() for k, v in rows.items()}


def fit_threshold(r, y):
    """thresholds.fit_threshold plus the SE of the threshold c (inverse Hessian, delta method)."""
    if len(r) < 200 or y.mean() in (0, 1):
        return np.nan, np.nan, np.nan
    X = np.stack([np.ones_like(r), r - 1], 1)
    beta = np.array([0.0, 10.0])
    for _ in range(50):
        z = X @ beta
        pz = 1 / (1 + np.exp(-z))
        g = X.T @ (y - pz)
        Hm = -(X * (pz * (1 - pz))[:, None]).T @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        beta = beta - step
        if np.abs(step).max() < 1e-8:
            break
    pz = 1 / (1 + np.exp(-(X @ beta)))
    cov = np.linalg.inv((X * (pz * (1 - pz))[:, None]).T @ X + 1e-6 * np.eye(2))
    a, b = beta
    grad = np.array([-1 / b, a / b**2])
    return 1 - a / b, 1 / abs(b), float(np.sqrt(grad @ cov @ grad))


def sec_K():
    """Claim 11: new-max comparison threshold by age of the stored max (thresholds.py)."""
    t0 = time.time()
    names, true_acc, models = bench_all()
    sel = ["example"] + [m for m in names if m.startswith("mult")]
    n, seed = 2**23, 1111
    tab = {}
    for m in sel:
        v = verdicts(models[m], n, seed)
        tab[m] = {}
        for kind in ("max", "2nd"):
            d = v[kind]
            for age in range(9):
                s = d[:, 2] == age
                c, w, se = fit_threshold(d[s, 0], d[s, 1])
                tab[m][f"{kind}_{age}"] = dict(c=c, w=w, se=se, n=int(s.sum()))
    base = [tab["example"][f"max_{a}"] for a in range(9)]
    print("  base new-max threshold by age: " + " ".join(f"{b['c']:.3f}±{b['se']:.3f}" for b in base))
    allfits = [(m, a, tab[m][f"max_{a}"]) for m in sel[1:] for a in range(9)
               if np.isfinite(tab[m][f"max_{a}"]["c"])]
    good = [r for r in allfits if r[2]["se"] <= 0.02]           # well-determined fits only
    hi = max(good, key=lambda r: r[2]["c"])
    lo = min(good, key=lambda r: r[2]["c"])
    raw_hi = max(allfits, key=lambda r: r[2]["c"])
    raw_lo = min(allfits, key=lambda r: r[2]["c"])
    n_out = len({r[0] for r in good if abs(r[2]["c"] - 1) > 0.1})
    bc = [b["c"] for b in base if np.isfinite(b["c"])]
    ok_base = max(abs(c - 1) for c in bc) <= 0.025

    def traj(m):
        return ", ".join(f"{tab[m][f'max_{a}']['c']:.2f}" for a in range(9))

    rows = [
        row("11a", "M16,10's new-max comparison is nearly unbiased at every age: the ratio x / stored max at which the "
                   "network is equally likely to treat a new input as a new max or as a new 2nd, for stored maxima "
                   "0-8 steps old.",
            f"{min(bc):.3f}-{max(bc):.3f} over ages 0-{len(bc) - 1}",
            f"{n:,} random inputs; {min(b['n'] for b in base):,}-{max(b['n'] for b in base):,} near-tie readouts "
            "per age, steps 5-10",
            f"fit SE {min(b['se'] for b in base):.4f}-{max(b['se'] for b in base):.4f}",
            "reproduced" if ok_base else "changed", "1.00 +- 0.02 at every age",
            "thresholds.py: logistic fit of the pointer's verdict (old max's slot vs newest slot) on x/max, using "
            "ratios within 0.15 of 1. Seed 1111."),
        row("11b", "In noisy copies of M16,10 the new-max threshold drifts with the age of the stored max: highest "
                   "and lowest well-determined threshold over the 96 copies and ages 0-8.",
            f"highest {hi[2]['c']:.2f} ({hi[0]}, max {hi[1]} steps old); lowest {lo[2]['c']:.2f} ({lo[0]}, "
            f"{lo[1]} steps old), both outside the fitted ratios 0.85-1.15 and so extrapolated; {n_out} of 96 copies "
            "move more than 0.1 from 1 at some age",
            f"{len(good)} of {len(allfits)} fits (96 copies x 9 ages) with SE <= 0.02; {n:,} inputs per copy",
            f"fit SE {hi[2]['se']:.3f} and {lo[2]['se']:.3f}", "changed",
            "climbs to 1.72 or sinks to 0.68 (copies not named)",
            f"As 11a. The old 1.72 matches {raw_hi[0]} at age {raw_hi[1]} ({raw_hi[2]['c']:.2f}, SE "
            f"{raw_hi[2]['se']:.2f}), an extrapolation far outside the fitted ratios 0.85-1.15; its thresholds by "
            f"age 0-8 are {traj(raw_hi[0])}."),
    ]
    det = dict(n=n, seed=seed, table=tab)
    save("K", "Comparison thresholds by age", rows, det, time.time() - t0,
         f"Thresholds beyond about 0.85 or 1.15 lie outside the ratios used in the fit and are extrapolations. "
         f"The unfiltered extremes are {raw_hi[2]['c']:.2f} ({raw_hi[0]}, age {raw_hi[1]}, SE {raw_hi[2]['se']:.2f}) "
         f"and {raw_lo[2]['c']:.2f} ({raw_lo[0]}, age {raw_lo[1]}, SE {raw_lo[2]['se']:.2f}). {raw_lo[0]}'s "
         f"thresholds by age 0-8: {traj(raw_lo[0])}. The base model's thresholds by age: "
         + ", ".join(f"{c:.3f}" for c in bc) + ".")


# ----------------------------------------------------------------------------------------------- L
def fidelity_counts(params, n, seed, t_min=5, chunk=2**20):
    """fidelity.fidelity, vectorized with bincount (same definition). Returns good, total over
    key = event*100 + age1*10 + age2, events 0 ordinary, 1 new 2nd, 2 new max."""
    tp = tparams(params, th.float64)
    w_oh = tp[2]
    good = th.zeros(300, dtype=th.float64, device=DEV)
    tot = th.zeros(300, dtype=th.float64, device=DEV)
    for x in xstream(n, seed, chunk, th.float64):
        H = run_states(tp, x)
        for t in range(t_min, 10):
            pre = x[:, :t]
            o = pre.argsort(1, descending=True)
            v = pre.gather(1, o)
            m1, m2, p1, p2 = v[:, 0], v[:, 1], o[:, 0], o[:, 1]
            ptr0 = (H[t] @ w_oh.T).argmax(1)
            ptr1 = (H[t + 1] @ w_oh.T).argmax(1)
            right0 = ptr0 == p2 + (10 - t)
            xt = x[:, t]
            ok_pos = m2 > 0
            evs = [ok_pos & (xt < 0.8 * m2), ok_pos & (xt > 1.2 * m2) & (xt < 0.8 * m1), ok_pos & (xt > 1.2 * m1)]
            wants = [p2 + (10 - t - 1), th.full_like(p2, 9), p1 + (10 - t - 1)]
            age1 = (t - 1) - p1
            age2 = (t - 1) - p2
            for e in range(3):
                s = evs[e] & right0
                key = e * 100 + age1 * 10 + age2
                tot += th.bincount(key[s], minlength=300).double()
                good += th.bincount(key[s], weights=(ptr1 == wants[e])[s].double(), minlength=300)
        del H
    return good.cpu().numpy().reshape(3, 10, 10), tot.cpu().numpy().reshape(3, 10, 10)


def sec_L():
    """Claim 12: pointer fidelity on clear-cut ordinary steps, by age of the 2nd max (fidelity.py)."""
    t0 = time.time()
    names, true_acc, models = bench_all()
    sel = ["example"] + [m for m in names if m.startswith("mult")]
    n, seed = 2**22, 1212
    res = {}
    for m in sel:
        g, tt = fidelity_counts(models[m], n, seed)
        by2_g, by2_n = g.sum(1), tt.sum(1)          # [event, age2]
        res[m] = dict(fid=by2_g / np.maximum(by2_n, 1), n=by2_n)
    b = res["example"]
    ages = [a for a in range(9) if b["n"][0, a] > 0]
    bf = [b["fid"][0, a] for a in ages]
    bse_ = [bse(b["fid"][0, a], b["n"][0, a]) for a in ages]
    print("  base ordinary fidelity by age of 2nd: " + " ".join(f"{a}:{f:.4f}" for a, f in zip(ages, bf)))
    # copies with the lowest ordinary fidelity at ages 5-8 of the 2nd max
    score = {m: res[m]["fid"][0, 5:9].mean() for m in sel[1:]}
    worst = sorted(score, key=score.get)[:3]
    for m in worst + DEGRADED4:
        f = res[m]["fid"][0, 5:9]
        print(f"  {m:14s} ordinary fidelity at 2nd-max ages 5-8: " + " ".join(f"{v:.3f}" for v in f)
              + f"  (n {res[m]['n'][0, 5:9].astype(int).tolist()}, true acc {true_acc[m]:.3f})")
    w = worst[0]
    wf = res[w]["fid"][0, 5:9]
    wn = res[w]["n"][0, 5:9]
    least = min(sel[1:], key=lambda m: true_acc[m])
    least_txt = " (also the least accurate copy)" if least == w else ""
    # the same copy on fidelity.py's own inputs (one draw of 2 x 10^6, seed 0)
    g0, t0_ = fidelity_counts(models[w], 2 * 10**6, 0, chunk=2 * 10**6)
    orig = (g0.sum(1) / np.maximum(t0_.sum(1), 1))[0, 5:9]
    ok_b = abs(min(bf) - 0.98) <= 0.01 and abs(max(bf) - 0.995) <= 0.005
    ok_w = abs(wf.min() - 0.16) <= 0.03 and abs(wf.max() - 0.34) <= 0.03
    rows = [
        row("12a", "On clear-cut ordinary steps (new input below 0.8 x the running 2nd max), M16,10's running answer "
                   "stays on the 2nd max this often, by age of the 2nd max.",
            f"{pct(min(bf))}-{pct(max(bf))} over ages {ages[0]}-{ages[-1]}",
            f"{n:,} random inputs; {int(min(b['n'][0, a] for a in ages)):,}-"
            f"{int(max(b['n'][0, a] for a in ages)):,} qualifying steps per age, steps 5-9",
            f"SE {pct(min(bse_), 2)}-{pct(max(bse_), 2)}", "reproduced" if ok_b else "changed", "98-99.5%",
            "fidelity.py: running answer argmax(W_oh h_t) right before the step and running 2nd max > 0; count "
            "steps where it shifts with the 2nd max. Seed 1212."),
        row("12b", f"On the same steps, the noisy copy of M16,10 with the lowest fidelity at these ages{least_txt} "
                   "loses its running answer when the 2nd max is 5-8 steps old: the share of steps it keeps it, at "
                   "ages 5, 6, 7 and 8.",
            f"{w} (true accuracy {true_acc[w]:.3f}): " + ", ".join(pct(v, 0) for v in wf),
            "qualifying steps per age: " + ", ".join(f"{int(v):,}" for v in wn),
            "SE " + ", ".join(pct(bse(f, k), 2) for f, k in zip(wf, wn)),
            "reproduced" if ok_w else "changed", "16-34% (copy not named)",
            f"As 12a, all 96 copies; {w} has the lowest mean over ages 5-8. On fidelity.py's own inputs it gives "
            + ", ".join(pct(v, 0) for v in orig) + ": the old range quoted only ages 5 and 8."),
    ]
    det = dict(n=n, seed=seed, per_model={m: dict(ordinary=res[m]["fid"][0], ordinary_n=res[m]["n"][0],
                                                  new2=res[m]["fid"][1], newmax=res[m]["fid"][2])
                                          for m in sel})
    save("L", "Pointer fidelity", rows, det, time.time() - t0)


# ----------------------------------------------------------------------------------------------- M
def sec_M():
    """Claim 13: zero one comparator neuron in the final state h_10, accuracy by type of the last input."""
    t0 = time.time()
    tp = tparams(example())
    w_oh = tp[2]
    n, seed = 2**24, 1313
    neurons = [None, 7, 6, 1, 8, 4, 2]
    types = ["new max", "new 2nd", "new 3rd", "low"]
    hit = np.zeros((len(neurons), 4))
    cnt = np.zeros(4)
    for x in xstream(n, seed, 2**21):
        h = run_states(tp, x)[-1]
        y = target(x)
        v = x[:, :9].sort(1, descending=True).values
        x9 = x[:, 9]
        typ = th.where(x9 > v[:, 0], 0, th.where(x9 > v[:, 1], 1, th.where(x9 > v[:, 2], 2, 3)))
        cnt += th.bincount(typ, minlength=4).cpu().numpy()
        for i, j in enumerate(neurons):
            hz = h.clone()
            if j is not None:
                hz[:, j] = 0
            c = ((hz @ w_oh.T).argmax(-1) == y).double()
            hit[i] += th.bincount(typ, weights=c, minlength=4).cpu().numpy()
    acc = hit / cnt
    for i, j in enumerate(neurons):
        print(f"  zero {'none' if j is None else 'n' + str(j):5s}: " + " ".join(f"{t} {a:.3f}" for t, a in zip(types, acc[i])))
    comp = acc[1:5][:, 2:4]                     # n7, n6, n1, n8 x new 3rd, low
    a4 = acc[5, 0]
    ok = abs(comp.min() - 0.20) <= 0.03 and abs(comp.max() - 0.43) <= 0.03
    rows = [
        row("13a", "M16,10's comparators are causal: zeroing any one of n7, n6, n1 or n8 in the final state drops "
                   "accuracy on inputs whose last value is low (below the 3rd largest of the first nine) or a new "
                   "3rd to this range.",
            f"{pct(comp.min(), 0)}-{pct(comp.max(), 0)} (unablated: new 3rd {pct(acc[0, 2])}, low {pct(acc[0, 3])})",
            f"{n:,} random inputs: {int(cnt[2]):,} with a new-3rd last input, {int(cnt[3]):,} with a low one",
            f"SE {pct(max(bse(a, cnt[2 + k % 2]) for k, a in enumerate(comp.ravel())), 2)} or less",
            "reproduced" if ok else "changed", "20-43%",
            "Set h_10[j] = 0 before the readout; last-input type relative to the max, 2nd and 3rd of x_0..x_8. "
            "No original script; natural definition. Seed 1313. Per neuron (new 3rd / low): "
            + "; ".join(f"n{j} {pct(acc[i + 1, 2], 0)} / {pct(acc[i + 1, 3], 0)}" for i, j in enumerate([7, 6, 1, 8]))
            + "."),
        row("13b", "Zeroing the max-jump neuron n4 in M16,10's final state breaks new maxima: accuracy on inputs whose "
                   "last value is a new max.",
            f"{pct(a4)} (unablated {pct(acc[0, 0])})", f"{int(cnt[0]):,} inputs with a new-max last input",
            f"SE {pct(bse(a4, cnt[0]), 2)}", "reproduced" if agree(a4, 0.52, bse(a4, cnt[0]), 0.01) else "changed",
            "52%", "As 13a."),
    ]
    det = dict(n=n, seed=seed, neurons=["none" if j is None else j for j in neurons], types=types, acc=acc,
               count=cnt)
    save("M", "Comparator zeroing", rows, det, time.time() - t0)


# ----------------------------------------------------------------------------------------------- N
def sec_N():
    """Claim 14: exact length-3 accuracy (exact_s2.py) against 2^30-input Monte Carlo."""
    t0 = time.time()
    old = {2: (0.4101721, 0.4101804, 1.6e-5, 18), 4: (0.9845503, 0.9845533, 3.9e-6, 100),
           16: (0.9995044, 0.9995047, 7.0e-7, 926), 32: (0.9995628, 0.9995636, 6.6e-7, 3174)}
    res = {}
    for i, d in enumerate((2, 4, 16, 32)):
        p = load(f"2nd_argmax_{d}_3_1073741824_bo5.pth")
        te = time.time()
        ex, ops, nreg = exact_s2.exact_accuracy(*p)
        te = time.time() - te
        mc, nn = accuracy(p, 2**30, seed=1400 + i)
        se = bse(mc, nn)
        res[d] = dict(exact=ex, ops=ops, regions=nreg, mc=mc, mc_se=se, z=(ex - mc) / se, n=nn, exact_seconds=te)
        print(f"  M{d},3: exact {ex:.7f} ({nreg} regions, {ops:.2g} ops, {te:.0f}s); MC {mc:.7f} ± {se:.1e};"
              f" z {(ex - mc) / se:+.2f}")
    rows = []
    for d, rid in zip((2, 4, 16, 32), "abcd"):
        r = res[d]
        o = old[d]
        same = abs(r["exact"] - o[0]) < 1e-7 and r["regions"] == o[3] and abs(r["z"]) < 3
        rows.append(row(
            f"14{rid}", f"Exact, sampling-free accuracy of the length-3 2nd-argmax model M{d},3 (ARC's best-of-5, "
                        f"{d} neurons) matches Monte Carlo.",
            f"exact {r['exact']:.7f}; Monte Carlo {r['mc']:.7f} (difference {(r['exact'] - r['mc']) / r['mc_se']:+.1f} SE)",
            f"exact: {r['regions']:,} linear regions, {r['ops']:,.0f} multiply-adds; Monte Carlo: {r['n']:,} inputs",
            f"Monte Carlo SE {r['mc_se']:.1e}", "reproduced" if same else "changed",
            f"exact {o[0]:.7f}; Monte Carlo {o[1]:.7f} +- {o[2]:.1e}; {o[3]:,} regions",
            f"exact_s2.exact_accuracy (spherical polygons, first-index tie rule); core.accuracy float32, seed "
            f"{1400 + (2, 4, 16, 32).index(d)}."))
    save("N", "Exact length-3 table", rows, dict(results=res), time.time() - t0)


# ----------------------------------------------------------------------------------------------- markdown
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_md():
    data = json.loads(JSON_PATH.read_text())
    keys = [k for k in "ABCDEFGHIJKLMN" if k in data]
    rows = [r for k in keys for r in data[k]["rows"]]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    missing = [k for k in "ABCDEFGHIJKLMN" if k not in data]
    lines = [
        "# M16,10 claims, re-run with fixed seeds",
        "",
        f"Status of the {len(rows)} table rows: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + "."
        + (f" Sections not run: {', '.join(missing)}." if missing else ""),
        "",
        "Every row was re-run on " + ", ".join(sorted({data[k]['run_at'][:10] for k in keys}))
        + " by `scripts/claims.py` (one section per claim, `python claims.py <letter>`), with the seed named in "
          "the row. Raw numbers are in `claims.json`. Old values come from `20261016_m16_10/REPORT.md`, `NOTE.md` and "
          "`reviews/review_1.md`.",
        "",
        "M16,10 is ARC's example 2nd-argmax RNN: 16 ReLU neurons, no biases, reading 10 standard Gaussian inputs. "
        "Its target is the position of the 2nd-largest input. Noisy copies are the 96 models in `benchmark.npz`, "
        "each weight multiplied by 1 + sigma N(0,1) for sigma = 0.005, 0.01 and 0.02 (32 each), with true "
        "accuracies from 2^29 inputs.",
        "",
        "Status key. **Reproduced**: the new value matches the old one at the old value's stated precision (within "
        "3 standard errors plus half a unit of its last digit; for ranges, each end within 0.03). **Changed**: the "
        "new value differs; quote the new one. **Not reproducible**: the claim could not be re-run; the row says why. "
        "SE is a standard error from the stated sample; intervals over copies are 95% bootstrap intervals.",
        "",
        "| Claim | Value | n (what is counted) | SE or interval | Status | Old value | How measured |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(esc(x) for x in (f"**{r['id']}.** {r['claim']}", r["value"], r["n"], r["se"],
                                                         r["status"], r["old"], r["how"])) + " |")
    notes = [(k, data[k]["notes"]) for k in keys if data[k].get("notes")]
    if notes:
        lines += ["", "## Notes", ""]
        for k, t in notes:
            lines.append(f"- **{data[k]['title']}.** {t}")
    extra = MD_EXTRA.strip()
    if extra:
        lines += ["", extra]
    MD_PATH.write_text("\n".join(lines) + "\n")
    print("wrote", MD_PATH)


MD_EXTRA = ""

SECTIONS = dict(A=sec_A, B=sec_B, C=sec_C, D=sec_D, E=sec_E, F=sec_F, G=sec_G, H=sec_H, I=sec_I, J=sec_J,
                K=sec_K, L=sec_L, M=sec_M, N=sec_N)

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        args = list(SECTIONS)
    for a in args:
        if a == "md":
            continue
        print(f"== section {a}: {SECTIONS[a].__doc__.strip().splitlines()[0]}", flush=True)
        t = time.time()
        with th.no_grad() if a != "I" else th.enable_grad():
            SECTIONS[a]()
        th.cuda.empty_cache()
        print(f"   ({time.time() - t:.0f}s, peak GPU {th.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
        th.cuda.reset_peak_memory_stats()
    if JSON_PATH.exists():
        write_md()
