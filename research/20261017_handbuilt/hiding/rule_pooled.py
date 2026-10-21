"""Simpler written rule: as rule_kernel.py (mode 'lag': shared kernels by lag for the numbers since the record and for
the echo of clipped amounts), but the base reference no longer depends on where the record sits, only on K, the steps
since it: c_n ref_n = a_n(K) P + b_n(K) M + sum_d g_n(K, d) P [P arrived d steps before the record] + kernels.
Variants: with the P-arrival terms, and without them (P's coefficient then depends on K only). Before the first
positive number (r = -1) the reference is fitted per step on nothing but the numbers so far (kernels). Fit seed 3,
score seed 4, 200,000 lists each."""
import numpy as np
from rulelib import list_vars, lists, run, report, C, WI, WH, WCC, CN

NL = 9


def regime_key(t, rr):
    return ("K", t - rr) if rr >= 0 else ("neg", t)


def design(X, V, t, rr, idx, Hc, E, age):
    if rr >= 0:
        reg = [V["P"][idx, t], V["M"][idx, t]]
        if age:
            dd = rr - V["RP"][idx, t]
            reg += [V["P"][idx, t] * (dd == d) for d in range(1, 10)]
    else:
        reg = [np.zeros(len(idx))]
    reg = np.stack(reg, 1)
    sh = np.zeros((len(idx), 2 * NL + 4 * NL))
    for j in range(rr + 1, t):
        l = t - j; sh[:, 2 * (l - 1)] = X[idx, j]; sh[:, 2 * (l - 1) + 1] = X[idx, j] * V["KEPT"][idx, j]
    if rr >= 0:
        for j in range(rr, t):
            l = t - j; src = Hc[idx, rr] if j == rr else E[idx, j]
            sh[:, 2 * NL + 4 * (l - 1): 2 * NL + 4 * l] = src
    return reg, sh


def fit(X, age):
    V = list_vars(X); _, P, H = run(X); Hc = H[:, :, C]; E = np.maximum(-P[:, :, C], 0)
    target = P[:, :, C] - X[:, :, None] * WI[C]
    keys = sorted({regime_key(t, rr) for t in range(1, 10) for rr in range(-1, t)}, key=str)
    nreg = (11 if age else 2); offs = {k: (i * nreg, nreg if k[0] == "K" else 1) for i, k in enumerate(keys)}
    o = len(keys) * nreg; nsh = 6 * NL; ntot = o + nsh
    AtA = np.zeros((ntot, ntot)); Aty = np.zeros((ntot, 4))
    for t in range(1, 10):
        for rr in range(-1, t):
            idx = np.where(V["R"][:, t] == rr)[0]
            if len(idx) == 0: continue
            reg, sh = design(X, V, t, rr, idx, Hc, E, age); a, n = offs[regime_key(t, rr)]
            cols = np.r_[np.arange(a, a + n), np.arange(o, ntot)]; A = np.c_[reg, sh]
            AtA[np.ix_(cols, cols)] += A.T @ A; Aty[cols] += A.T @ target[idx, t]
    sol = np.linalg.solve(AtA + 1e-6 * np.eye(ntot), Aty)
    return {"sol": sol, "offs": offs, "o": o, "age": age}


def substitute(X, m):
    V = list_vars(X); B = len(X); Hc = np.zeros((B, 10, 4)); E = np.zeros((B, 10, 4)); sol, offs, o = m["sol"], m["offs"], m["o"]
    def hook(t, pre, hn, hprev):
        cref = np.zeros((B, 4))
        if t > 0:
            for rr in range(-1, t):
                idx = np.where(V["R"][:, t] == rr)[0]
                if len(idx) == 0: continue
                reg, sh = design(X, V, t, rr, idx, Hc, E, m["age"]); a, n = offs[regime_key(t, rr)]
                cref[idx] = reg @ sol[a:a + n] + sh @ sol[o:]
        prer = cref - CN * X[:, t:t + 1]
        hn = hn.copy(); hn[:, C] = np.maximum(prer, 0); Hc[:, t] = hn[:, C]; E[:, t] = np.maximum(-prer, 0)
        return hn
    return run(X, hook=hook)


if __name__ == "__main__":
    Xf, Xs = lists(3, 200_000), lists(4, 200_000)
    report("network", Xs, run(Xs)[0].argmax(1))
    for age in (True, False):
        m = fit(Xf, age); report(f"pooled over record position, P-arrival terms {age}", Xs, substitute(Xs, m)[0].argmax(1))
        if age: np.save("rule_pooled_sol.npy", m["sol"])
    # robustness: other scoring lists
    m = fit(Xf, True); Xs2 = lists(5, 200_000)
    report("network (seed 5)", Xs2, run(Xs2)[0].argmax(1)); report("pooled, P-arrival terms (seed 5)", Xs2, substitute(Xs2, m)[0].argmax(1))
