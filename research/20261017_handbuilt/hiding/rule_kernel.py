"""Rule with shared kernels (the written form). c_n ref_n(t) =
   per-regime table:  a_n(t, r) P + b_n(t, r) M + sum_d g_n(t, r, d) P [P arrived at d]
 + numbers since, by lag l = t - j:  u_n(l) x_j + v_n(l) x_j [x_j kept]
 + echo, by lag l = t - j:  sum_i G_ni(l) eps_i(j), with eps(r) = the four's leftover at the record step and
   eps(j) = their clipped amounts at later steps (both are departures of the four's state from the rule's linear
   continuation, so the same kernel carries them).
The tables are fitted per regime, the kernels jointly over all regimes (normal equations, one least squares per
neuron). Fit seed 3 (200,000 lists), score seed 4. Variants: kernels by lag only; kernels by lag and K; echo kernel
fixed to the four's own block W_cc^l (no fit)."""
import numpy as np, json, sys
from rulelib import list_vars, lists, run, report, C, WI, WH, WCC, CN
from common import kinds

NL = 9


def design(X, V, t, rr, idx, Hc, E, mode):
    """Returns (regime columns, shared columns). Shared columns are laid out by lag (and K if mode == 'lagK')."""
    reg = [V["P"][idx, t], V["M"][idx, t]] + [V["P"][idx, t] * (V["RP"][idx, t] == d) for d in range(-1, max(rr, 0))]
    reg = np.stack(reg, 1)
    K = t - rr if rr >= 0 else 0
    nsh = (2 * NL + 4 * NL) * (10 if mode == "lagK" else 1)
    sh = np.zeros((len(idx), nsh)); base = (2 * NL + 4 * NL) * (K if mode == "lagK" else 0)
    for j in range(rr + 1, t):
        l = t - j
        sh[:, base + 2 * (l - 1)] = X[idx, j]; sh[:, base + 2 * (l - 1) + 1] = X[idx, j] * V["KEPT"][idx, j]
    if mode != "fixed_echo" and rr >= 0:
        for j in range(rr, t):
            l = t - j; src = Hc[idx, rr] if j == rr else E[idx, j]
            sh[:, base + 2 * NL + 4 * (l - 1): base + 2 * NL + 4 * l] = src
    return reg, sh


def fixed_echo(t, rr, idx, Hc, E):
    out = np.zeros((len(idx), 4))
    if rr < 0: return out
    for j in range(rr, t):
        src = Hc[idx, rr] if j == rr else E[idx, j]
        out += src @ np.linalg.matrix_power(WCC, t - j).T
    return out


def fit(X, mode):
    V = list_vars(X); _, P, H = run(X); Hc = H[:, :, C]; E = np.maximum(-P[:, :, C], 0)
    target = P[:, :, C] - X[:, :, None] * WI[C]
    regs = [(t, rr) for t in range(1, 10) for rr in range(-1, t)]
    blocks = {}; offs = {}; o = 0
    for (t, rr) in regs:
        nreg = 2 + max(rr, 0) + 1; offs[(t, rr)] = (o, nreg); o += nreg
    nsh = (2 * NL + 4 * NL) * (10 if mode == "lagK" else 1); ntot = o + nsh
    AtA = np.zeros((ntot, ntot)); Aty = np.zeros((ntot, 4))
    for (t, rr) in regs:
        idx = np.where(V["R"][:, t] == rr)[0]
        if len(idx) == 0: continue
        reg, sh = design(X, V, t, rr, idx, Hc, E, mode)
        y = target[idx, t] - (fixed_echo(t, rr, idx, Hc, E) if mode == "fixed_echo" else 0)
        a, n = offs[(t, rr)]; cols = np.r_[np.arange(a, a + n), np.arange(o, ntot)]
        A = np.c_[reg, sh]; AtA[np.ix_(cols, cols)] += A.T @ A; Aty[cols] += A.T @ y
    AtA += 1e-6 * np.eye(ntot)
    sol = np.linalg.solve(AtA, Aty)
    return {"sol": sol, "offs": offs, "o": o, "mode": mode}


def substitute(X, model):
    V = list_vars(X); B = len(X); Hc = np.zeros((B, 10, 4)); E = np.zeros((B, 10, 4)); mode = model["mode"]
    sol, offs, o = model["sol"], model["offs"], model["o"]
    def hook(t, pre, hn, hprev):
        cref = np.zeros((B, 4))
        if t > 0:
            for rr in range(-1, t):
                idx = np.where(V["R"][:, t] == rr)[0]
                if len(idx) == 0: continue
                reg, sh = design(X, V, t, rr, idx, Hc, E, mode); a, n = offs[(t, rr)]
                cref[idx] = reg @ sol[a:a + n] + sh @ sol[o:]
                if mode == "fixed_echo": cref[idx] += fixed_echo(t, rr, idx, Hc, E)
        prer = cref - CN * X[:, t:t + 1]
        hn = hn.copy(); hn[:, C] = np.maximum(prer, 0); Hc[:, t] = hn[:, C]; E[:, t] = np.maximum(-prer, 0)
        return hn
    return run(X, hook=hook)


if __name__ == "__main__":
    Xf, Xs = lists(3, 200_000), lists(4, 200_000)
    report("network", Xs, run(Xs)[0].argmax(1))
    for mode in ("lag", "lagK", "fixed_echo"):
        m = fit(Xf, mode); L, Pr, Hs = substitute(Xs, m)
        report(f"shared kernels, mode {mode}", Xs, L.argmax(1))
        if mode == "lag":
            np.savez("rule_kernel_lag.npz", sol=m["sol"], o=m["o"], offs=json.dumps({f"{k[0]},{k[1]}": v for k, v in m["offs"].items()}))
