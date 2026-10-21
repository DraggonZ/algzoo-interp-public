"""Forward propagation of a probabilistic model of the hidden state (Hilton's proposal), version 2.

The distribution over z = (state h [16], largest so far L, second-largest so far S) is a mixture of Gaussians, one per
regime. Regime = positions of the largest and second-largest so far (they fix the answer, and the ages and 'which of
the largest have arrived' that the rules use), optionally refined by comparator outcomes.

One step, per component:
  1. append the new number x ~ N(0,1) (independent of z) and, if weight noise is modelled, one noise term per neuron;
  2. split by list events: new largest (x > L), new second (S < x < L), neither (x < S); optional comparator splits
     x > c L or x > c S for given (neuron-rule) ratios c; each piece is the Gaussian truncated to the half-spaces,
     moment matched;
  3. optionally split on a neuron's own sign (u_j > 0 vs < 0) where it is uncertain, at most once per neuron per step;
     in a split piece the neuron is exactly linear (on or off);
  4. every other neuron passes its Gaussian input u_j through the ReLU with exact mean and variance; covariances use
     the gain P(u_j > 0) (exact against linear functions of the inputs, Stein's lemma; approximate between two ReLUs);
  5. merge pieces with the same regime by moment matching.
End: per component, the probability that the second-largest's score beats the other nine (Gaussian orthant), weighted.

Gap as noise: var_dw gives each weight's variance; the neuron's input gets independent zero-mean noise of variance
sum_k var(dW_jk) E[y_k^2] (fresh each step), and the scores likewise."""
import numpy as np
from scipy.stats import norm, multivariate_normal

D = 18
from scipy.stats import qmc
_Z = norm.ppf(np.clip(qmc.Sobol(9, scramble=True, seed=0).random(2 ** 16), 1e-12, 1 - 1e-12))   # fixed quasi-random points for orthant probabilities


def trunc(mu, S, a, sign):
    """Condition N(mu, S) [K,d],[K,d,d] on sign * a.y > 0 (a [K,d]); returns prob, mu', S' (moment matched)."""
    m = np.einsum("kd,kd->k", a, mu); Sa = np.einsum("kde,ke->kd", S, a); v = np.einsum("kd,kd->k", a, Sa)
    det = v < 1e-14; sd = np.sqrt(np.where(det, 1.0, v)); alpha = -sign * m / sd
    p = norm.sf(alpha)
    lam = np.exp(norm.logpdf(alpha) - norm.logsf(alpha))
    vr = np.clip(1 + alpha * lam - lam ** 2, 0, 1)
    g = Sa / np.where(det, 1.0, v)[:, None]
    mu2 = mu + g * (sign * sd * lam)[:, None]
    S2 = S - np.einsum("kd,ke->kde", Sa, g) * (1 - vr)[:, None, None]
    p = np.where(det, (sign * m > 0).astype(float), p)
    mu2[det] = mu[det]; S2[det] = S[det]
    return p, mu2, S2


class Prop:
    def __init__(self, W, var_dw=0.0, split_neurons=(), split_band=(0.02, 0.98), comparators=(), keymask=(),
                 wmin=1e-9, verbose=False):
        """W = (w_in [16], W_hh [16,16], W_out [10,16]).
        split_neurons: neurons that may be split on their own sign when P(on) is inside split_band.
        comparators: list of ('L' or 'S', c) giving extra splits x > c*L or x > c*S (rules from the weights).
        keymask: neurons whose (split) on/off joins the regime key after the step."""
        self.wi, self.wh, self.wo = [np.asarray(w, float) for w in W]
        if np.isscalar(var_dw):
            var_dw = (np.full(16, var_dw), np.full((16, 16), var_dw), np.full((10, 16), var_dw))
        self.vi, self.vh, self.vo = [np.asarray(v, float) for v in var_dw]
        self.noisy = any(np.any(v > 0) for v in (self.vi, self.vh, self.vo))
        self.split = list(split_neurons); self.band = split_band; self.comp = list(comparators)
        self.keymask = list(keymask); self.wmin = wmin; self.verbose = verbose; self.log = []

    def run(self, T=10):
        comps = {(-1, -1, ()): (1.0, np.zeros(D), np.zeros((D, D)))}
        for t in range(T): comps = self.step(comps, t)
        self.comps = comps
        return self.accuracy(comps)

    def _cut(self, pieces, a_fn, sign_tags):
        """Split each piece by a half-space; a_fn(d) gives the direction; returns new list."""
        out = []
        for (w, mu, S, tags) in pieces:
            a = np.broadcast_to(a_fn(mu.shape[1]), mu.shape).copy()
            for sg, tg in sign_tags:
                p, m2, S2 = trunc(mu, S, a, sg)
                out.append((w * p, m2, S2, [t + (tg,) for t in tags]))
        return out

    def step(self, comps, t):
        keys = list(comps); K = len(keys)
        w = np.array([comps[k][0] for k in keys]); mu = np.stack([comps[k][1] for k in keys]); S = np.stack([comps[k][2] for k in keys])
        nn = 16 if self.noisy else 0; d = D + 1 + nn; ix = D; iL, iS = 16, 17
        muy = np.zeros((K, d)); Sy = np.zeros((K, d, d)); muy[:, :D] = mu; Sy[:, :D, :D] = S; Sy[:, ix, ix] = 1.0
        if nn:
            Eh2 = np.einsum("kii->ki", S[:, :16, :16]) + mu[:, :16] ** 2
            q = Eh2 @ self.vh.T + self.vi[None, :]
            for j in range(16): Sy[:, D + 1 + j, D + 1 + j] = q[:, j]
        posL = np.array([k[0] for k in keys]); posS = np.array([k[1] for k in keys])

        def vec(i, j=None, c=1.0):
            def f(dd):
                a = np.zeros(dd); a[i] = 1
                if j is not None: a[j] = -c
                return a
            return f
        # list events
        if t == 0:
            pieces = [(w, muy, Sy, [("R",)] * K)]
        else:
            pieces = self._cut([(w, muy, Sy, [()] * K)], vec(ix, iL), [(+1, "R"), (-1, "low")])
            rec = [pieces[0]]; low = pieces[1]
            if t == 1:
                low = (low[0], low[1], low[2], [("M",)] * K); pieces = rec + [low]
            else:
                mid_bot = self._cut([low], vec(ix, iS), [(+1, "M"), (-1, "B")])
                pieces = rec + [(p[0], p[1], p[2], [(tg[-1],) for tg in p[3]]) for p in mid_bot]
        # comparator splits x > c * L (or S)
        for (ref, c) in self.comp:
            if t == 0: continue
            pieces = self._cut(pieces, vec(ix, iL if ref == "L" else iS, c), [(+1, f"{ref}>{c}"), (-1, f"{ref}<{c}")])
        # flatten
        W_ = np.concatenate([p[0] for p in pieces]); MU = np.concatenate([p[1] for p in pieces]); SS = np.concatenate([p[2] for p in pieces])
        EV = np.concatenate([np.array([tg[0] for tg in p[3]]) for p in pieces])
        PL = np.concatenate([posL] * len(pieces)); PS = np.concatenate([posS] * len(pieces))
        ok = W_ > self.wmin; W_, MU, SS, EV, PL, PS = W_[ok], MU[ok], SS[ok], EV[ok], PL[ok], PS[ok]
        C = np.zeros((16, d)); C[:, :16] = self.wh; C[:, ix] = self.wi
        if nn: C[:, D + 1:] = np.eye(16)
        forced = np.full((len(W_), 16), -1)                     # -1 = ReLU by moments, 0/1 = split off/on
        nsplit = 0
        for j in self.split:
            m = MU @ C[j]; v = np.einsum("d,kde,e->k", C[j], SS, C[j])
            pon = np.where(v > 1e-14, norm.sf(-m / np.sqrt(np.maximum(v, 1e-300))), (m > 0).astype(float))
            amb = (pon > self.band[0]) & (pon < self.band[1])
            if not amb.any(): continue
            nsplit += amb.sum(); idx = np.where(amb)[0]; keep = ~amb
            a = np.broadcast_to(C[j], (len(idx), d)).copy()
            p1, m1, S1 = trunc(MU[idx], SS[idx], a, +1); p0, m0, S0 = trunc(MU[idx], SS[idx], a, -1)
            f1 = forced[idx].copy(); f1[:, j] = 1; f0 = forced[idx].copy(); f0[:, j] = 0
            W_ = np.concatenate([W_[keep], W_[idx] * p1, W_[idx] * p0]); MU = np.concatenate([MU[keep], m1, m0])
            SS = np.concatenate([SS[keep], S1, S0]); forced = np.concatenate([forced[keep], f1, f0])
            EV = np.concatenate([EV[keep], EV[idx], EV[idx]]); PL = np.concatenate([PL[keep], PL[idx], PL[idx]]); PS = np.concatenate([PS[keep], PS[idx], PS[idx]])
            ok = W_ > self.wmin; W_, MU, SS, forced, EV, PL, PS = W_[ok], MU[ok], SS[ok], forced[ok], EV[ok], PL[ok], PS[ok]
        n = len(W_)
        # pre-activations
        mu_u = MU @ C.T; S_u = np.einsum("jd,kde,ie->kji", C, SS, C); v_u = np.einsum("kjj->kj", S_u)
        s_u = np.sqrt(np.maximum(v_u, 1e-300)); r = mu_u / s_u
        Phi = np.where(v_u > 1e-14, norm.cdf(r), (mu_u > 0).astype(float)); phi = np.where(v_u > 1e-14, norm.pdf(r), 0.0)
        relu_m = np.where(v_u > 1e-14, mu_u * Phi + s_u * phi, np.maximum(mu_u, 0))
        relu_v = np.where(v_u > 1e-14, (mu_u ** 2 + v_u) * Phi + mu_u * s_u * phi - relu_m ** 2, 0.0)
        g = np.where(forced >= 0, forced.astype(float), Phi)                              # gains
        hm = np.where(forced >= 0, forced * mu_u, relu_m)
        hv = np.where(forced >= 0, forced * v_u, relu_v)
        # linear map for (h', L', S') from y with gains
        A = np.zeros((n, D, d)); A[:, :16, :] = g[:, :, None] * C[None]
        isR = EV == "R"; isM = EV == "M"
        A[:, iL, iL] = 1; A[:, iS, iS] = 1
        A[isR, iL, :] = 0; A[isR, iL, ix] = 1; A[isR, iS, :] = 0; A[isR, iS, iL] = 1
        A[isM, iS, :] = 0; A[isM, iS, ix] = 1
        MUn = np.einsum("kde,ke->kd", A, MU); MUn[:, :16] = hm
        SSn = np.einsum("kde,kef,kgf->kdg", A, SS, A)
        di = np.arange(16); SSn[:, di, di] = hv
        newL = np.where(isR, t, PL); newS = np.where(isR, PL, np.where(isM, t, PS))
        groups = {}
        for i in range(n):
            km = tuple(int(forced[i, j]) for j in self.keymask)
            groups.setdefault((int(newL[i]), int(newS[i]), km), []).append(i)
        out = {}
        for k, ii in groups.items():
            ii = np.array(ii); ww = W_[ii]; tot = ww.sum()
            m = (ww[:, None] * MUn[ii]).sum(0) / tot; dv = MUn[ii] - m
            Sg = (ww[:, None, None] * (SSn[ii] + dv[:, :, None] * dv[:, None, :])).sum(0) / tot
            out[k] = (tot, m, 0.5 * (Sg + Sg.T))
        self.log.append((t, len(out), n, int(nsplit), float(sum(v[0] for v in out.values()))))
        if self.verbose: print("step %d: %d regimes, %d pieces, %d splits, mass %.6f" % self.log[-1], flush=True)
        return out

    def score_dist(self, mu, S):
        h_m, h_S = mu[:16], S[:16, :16]
        sm = self.wo @ h_m; sS = self.wo @ h_S @ self.wo.T
        if self.noisy: sS = sS + np.diag(self.vo @ (np.diag(h_S) + h_m ** 2))
        return sm, sS

    def win(self, mu, S, k):
        """P(score k beats the other nine)."""
        sm, sS = self.score_dist(mu, S)
        others = [j for j in range(10) if j != k]
        A = np.zeros((9, 10)); A[np.arange(9), k] = 1; A[np.arange(9), others] = -1
        dm = A @ sm; dS = A @ sS @ A.T + 1e-9 * np.eye(9)
        Lc = np.linalg.cholesky(dS)
        return float(((dm[None] + _Z @ Lc.T) > 0).all(1).mean())

    def accuracy(self, comps):
        tot = 0.0; self.per = {}
        for k, (w, mu, S) in comps.items():
            p = self.win(mu, S, k[1]); self.per[k] = (w, p); tot += w * p
        return tot

    def mistakes(self):
        """Share of predicted mistakes naming the largest, and another position (third or background lumped)."""
        mL = mo = 0.0
        for k, (w, mu, S) in self.comps.items():
            pL = self.win(mu, S, k[0]); mL += w * pL
            mo += w * (1 - self.per[k][1] - pL)
        tot = mL + mo
        return mL / tot, mo / tot
