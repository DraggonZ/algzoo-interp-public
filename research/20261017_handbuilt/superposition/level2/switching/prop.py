"""Hilton-style forward propagation of a probabilistic model of the hidden state, with the switching kept.

The distribution over (state h [16], largest so far L, second-largest so far S) is a mixture of Gaussians. Each
component carries a regime: the positions of the largest and second-largest so far (which decide the answer and the
ages the rules talk about), optionally plus the on/off pattern. Per step:
  1. the new number x ~ N(0,1) is appended; the component is split by the list events 'x is a new largest'
     (x > L), 'x is a new second' (S < x < L), 'neither' (x < S); each piece is the Gaussian truncated to that
     region, moment matched (one half-space at a time);
  2. each neuron's input u_j = W_hh h + w_in x (+ noise, below) is linear in the piece's variables. Where its sign is
     certain (probability above 1 - EPS) the neuron is fixed on or off; otherwise the piece is split into u_j > 0 and
     u_j < 0, each truncated and moment matched. The rules say which neurons are uncertain where;
  3. with the mask fixed, the update is exactly linear: new mean and covariance;
  4. pieces with the same regime are merged by moment matching.
At the end: probability, per component, that the second-largest's score beats the other nine (Gaussian orthant
probability), summed with the component weights.

Discrepancy as noise: the weights used are W_ideal; the gap W_actual - W_ideal (or added weight noise) enters as
zero-mean noise on each neuron's input with variance sum_k var(dW_jk) E[y_k^2] (y = previous state and x), drawn fresh
each step (the same gap at every step is not modelled), and likewise on the scores. With var(dW) = 0 the
propagation uses the given weights exactly."""
import numpy as np
from scipy.stats import norm, multivariate_normal

EPS = 1e-6
D = 18           # h[0:16], L, S


def trunc(mu, S, a, c, sign):
    """Gaussian(s) mu [K,d], S [K,d,d] conditioned on sign*(a.y - c) > 0, a [K,d]. Returns prob, mu', S'."""
    m = np.einsum("kd,kd->k", a, mu) - c
    Sa = np.einsum("kde,ke->kd", S, a); v = np.einsum("kd,kd->k", a, Sa)
    sd = np.sqrt(np.maximum(v, 1e-300)); alpha = -sign * m / sd           # truncation point in standard units
    p = norm.sf(alpha); p = np.clip(p, 0, 1)
    lam = np.exp(norm.logpdf(alpha) - norm.logsf(alpha))                 # inverse Mills ratio
    lam = np.where(np.isfinite(lam), lam, np.maximum(alpha, 0))
    dm = sign * sd * lam                                                 # shift of the projection's mean
    vr = np.clip(1 + alpha * lam - lam ** 2, 0, 1)                       # variance ratio
    g = Sa / np.maximum(v, 1e-300)[:, None]
    mu2 = mu + g * dm[:, None]
    S2 = S - np.einsum("kd,ke->kde", Sa, g) * (1 - vr)[:, None, None]
    det = v < 1e-18                                                      # deterministic direction
    if det.any():
        p[det] = (sign * m[det] > 0).astype(float); mu2[det] = mu[det]; S2[det] = S[det]
    return p, mu2, S2


class Prop:
    def __init__(self, W, var_dw=0.0, pattern_key=(), eps=EPS, wmin=1e-9, verbose=False):
        """W = (w_in [16], W_hh [16,16], W_out [10,16]) as numpy; var_dw: scalar or (var_in [16], var_hh [16,16],
        var_out [10,16]) variance of the gap treated as noise; pattern_key: neurons whose on/off joins the regime key."""
        self.wi, self.wh, self.wo = [np.asarray(w, float) for w in W]
        if np.isscalar(var_dw):
            var_dw = (np.full(16, var_dw), np.full((16, 16), var_dw), np.full((10, 16), var_dw))
        self.vi, self.vh, self.vo = var_dw; self.noisy = any(np.any(v > 0) for v in var_dw)
        self.pkey = list(pattern_key); self.eps = eps; self.wmin = wmin; self.verbose = verbose; self.log = []

    def run(self, T=10):
        # components: dict key -> (w, mu [D], S [D,D]); key = (posL, posS, pattern bits)
        comps = {(-1, -1, ()): (1.0, np.zeros(D), np.zeros((D, D)))}
        for t in range(T):
            comps = self.step(comps, t)
        self.comps = comps
        return self.accuracy(comps)

    def step(self, comps, t):
        keys = list(comps); K = len(keys)
        w = np.array([comps[k][0] for k in keys]); mu = np.stack([comps[k][1] for k in keys]); S = np.stack([comps[k][2] for k in keys])
        # augmented y = [h, L, S, x, n(16)]
        nn = 16 if self.noisy else 0; d = D + 1 + nn; ix = D
        muy = np.zeros((K, d)); Sy = np.zeros((K, d, d)); muy[:, :D] = mu; Sy[:, :D, :D] = S; Sy[:, ix, ix] = 1.0
        if nn:
            Eh2 = np.einsum("kii->ki", S[:, :16, :16]) + mu[:, :16] ** 2               # E[h_k^2]
            q = Eh2 @ self.vh.T + self.vi[None, :] * 1.0                                 # E[x^2] = 1
            for j in range(16): Sy[:, D + 1 + j, D + 1 + j] = q[:, j]
        posL = np.array([k[0] for k in keys]); posS = np.array([k[1] for k in keys])
        pieces = []   # (weight, mu, S, newposL, newposS, event)
        if t == 0:
            pieces.append((w, muy, Sy, np.zeros(K, int), np.full(K, -1), 0))
        else:
            a = np.zeros((K, d)); a[:, ix] = 1; a[:, D - 2] = -1                        # x - L
            p_rec, m_rec, S_rec = trunc(muy, Sy, a, 0.0, +1)
            pieces.append((w * p_rec, m_rec, S_rec, np.full(K, t), posL, 0))
            p_low, m_low, S_low = trunc(muy, Sy, a, 0.0, -1)
            if t == 1:
                pieces.append((w * p_low, m_low, S_low, posL, np.full(K, t), 1))
            else:
                b = np.zeros((K, d)); b[:, ix] = 1; b[:, D - 1] = -1                    # x - S
                p_mid, m_mid, S_mid = trunc(m_low, S_low, b, 0.0, +1)
                pieces.append((w * p_low * p_mid, m_mid, S_mid, posL, np.full(K, t), 1))
                p_bot, m_bot, S_bot = trunc(m_low, S_low, b, 0.0, -1)
                pieces.append((w * p_low * p_bot, m_bot, S_bot, posL, posS, 2))
        W_ = np.concatenate([p[0] for p in pieces]); MU = np.concatenate([p[1] for p in pieces]); SS = np.concatenate([p[2] for p in pieces])
        PL = np.concatenate([p[3] for p in pieces]); PS = np.concatenate([p[4] for p in pieces])
        ok = W_ > self.wmin; W_, MU, SS, PL, PS = W_[ok], MU[ok], SS[ok], PL[ok], PS[ok]
        # pre-activation u = C y
        C = np.zeros((16, d)); C[:, :16] = self.wh; C[:, ix] = self.wi
        if nn: C[:, D + 1:] = np.eye(16)
        mask = np.zeros((len(W_), 16), bool); nsplit = 0
        for j in range(16):
            a = np.broadcast_to(C[j], MU.shape).copy()
            m = MU @ C[j]; v = np.einsum("d,kde,e->k", C[j], SS, C[j])
            pon = np.where(v > 1e-18, norm.sf(-m / np.sqrt(np.maximum(v, 1e-300))), (m > 0).astype(float))
            sure_on = pon > 1 - self.eps; sure_off = pon < self.eps; amb = ~(sure_on | sure_off)
            mask[sure_on, j] = True
            if amb.any():
                nsplit += amb.sum(); idx = np.where(amb)[0]
                p1, m1, S1 = trunc(MU[idx], SS[idx], a[idx], 0.0, +1)
                p0, m0, S0 = trunc(MU[idx], SS[idx], a[idx], 0.0, -1)
                keep = ~amb
                mk1 = mask[idx].copy(); mk1[:, j] = True; mk0 = mask[idx].copy(); mk0[:, j] = False
                W_ = np.concatenate([W_[keep], W_[idx] * p1, W_[idx] * p0]); MU = np.concatenate([MU[keep], m1, m0])
                SS = np.concatenate([SS[keep], S1, S0]); mask = np.concatenate([mask[keep], mk1, mk0])
                PL = np.concatenate([PL[keep], PL[idx], PL[idx]]); PS = np.concatenate([PS[keep], PS[idx], PS[idx]])
                ok = W_ > self.wmin; W_, MU, SS, mask, PL, PS = W_[ok], MU[ok], SS[ok], mask[ok], PL[ok], PS[ok]
        # linear update to z' = [mask*u, L', S']
        n = len(W_); M = np.zeros((n, D, d))
        M[:, :16, :] = mask[:, :, None] * C[None]
        # L', S' depend on the event: record -> L'=x, S'=L; new second -> S'=x; else unchanged.
        rec = PL == t; sec = (PS == t) & ~rec
        M[:, 16, D - 2] = 1; M[:, 17, D - 1] = 1
        M[rec, 16, :] = 0; M[rec, 16, ix] = 1; M[rec, 17, :] = 0; M[rec, 17, D - 2] = 1
        M[sec, 17, :] = 0; M[sec, 17, ix] = 1
        MUn = np.einsum("kde,ke->kd", M, MU); SSn = np.einsum("kde,kef,kgf->kdg", M, SS, M)
        # merge by regime key
        out = {}
        pk = [tuple(mask[i, self.pkey].tolist()) for i in range(n)]
        groups = {}
        for i in range(n): groups.setdefault((int(PL[i]), int(PS[i]), pk[i]), []).append(i)
        for k, ii in groups.items():
            ii = np.array(ii); ww = W_[ii]; tot = ww.sum()
            if tot <= 0: continue
            m = (ww[:, None] * MUn[ii]).sum(0) / tot
            dv = MUn[ii] - m
            Sg = (ww[:, None, None] * (SSn[ii] + dv[:, :, None] * dv[:, None, :])).sum(0) / tot
            out[k] = (tot, m, 0.5 * (Sg + Sg.T))
        self.log.append((t, len(out), n, int(nsplit), float(sum(v[0] for v in out.values()))))
        if self.verbose: print("step %d: %d regimes, %d pieces, %d splits, mass %.6f" % self.log[-1], flush=True)
        return out

    def comp_win(self, mu, S, ps):
        """P(score[ps] > all other scores) for state ~ N(mu, S), with readout noise if any."""
        h_m, h_S = mu[:16], S[:16, :16]
        sm = self.wo @ h_m; sS = self.wo @ h_S @ self.wo.T
        if self.noisy:
            Eh2 = np.diag(h_S) + h_m ** 2; sS = sS + np.diag(self.vo @ Eh2)
        others = [k for k in range(10) if k != ps]
        A = np.zeros((9, 10)); A[np.arange(9), ps] = 1; A[np.arange(9), others] = -1
        dm = A @ sm; dS = A @ sS @ A.T + 1e-9 * np.eye(9)
        return multivariate_normal(mean=-dm, cov=dS, allow_singular=True).cdf(np.zeros(9), abseps=1e-6, releps=1e-5)

    def accuracy(self, comps):
        tot = 0.0; self.per = {}
        for k, (w, mu, S) in comps.items():
            p = self.comp_win(mu, S, k[1]); self.per[k] = (w, p); tot += w * p
        return tot
