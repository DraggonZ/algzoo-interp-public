"""Forward propagation of a probabilistic model of ARC's hidden state (Hilton's proposal). Final version (dense, batched).

Model. The list's running quantities are kept exact; the hidden state is Gaussian given them.
  Regime = (position of the largest so far, position of the second-largest so far, the gates' on/off for the next
  step). The first two fix the answer and every age the rules use. The gates (keep gate 5, late gate 9) switch on
  the previous number's comparison (gates/README.md); their on/off is decided when that number is read, at its exact
  value, from the sign of the gate's drive, and the gate is then exactly on or off (linear) at the next step.
  Values: L (largest so far) and S (second-largest so far) on a grid of G values (midpoints of G equal-probability
  cells of N(0,1)); the new number x on the same grid, so the list events (new largest, new second, neither) are
  exact and land on grid nodes; ties within a cell split half and half.
  At each (regime, L node, S node) the state h is one Gaussian.
One step: for every source node and every x node, u = W_hh h + w_in x is Gaussian; each neuron not forced by a
regime gets the ReLU's exact mean and variance (gain P(u_j > 0) for covariances, exact against linear functions,
approximate between two ReLUs); the result is added by moment matching to its destination node.
End: per node, P(the second-largest's score is highest) and P(the largest's score is highest) from the Gaussian
scores (fixed scrambled Sobol points), weighted by node probability.

Gap as noise (Hilton): var_dw = per-weight variances of the difference between the actual weights and the weights
used; each neuron's input gets zero-mean noise of variance sum_k var(dW_hh[j,k]) E[h_k^2] + var(dw_in[j]) x^2,
independent across neurons and fresh each step; the scores get sum_k var(dW_out[i,k]) E[h_k^2]."""
import numpy as np, torch
from scipy.stats import norm, qmc

DEV = "cuda"; dt = torch.float64


def relu_moments(m, v):
    s = v.clamp_min(1e-30).sqrt(); r = m / s
    Phi = torch.special.ndtr(r); phi = torch.exp(-0.5 * r * r) * 0.3989422804014327
    mean = m * Phi + s * phi; var = ((m * m + v) * Phi + m * s * phi - mean * mean).clamp_min(0)
    det = v < 1e-20
    return (torch.where(det, m.clamp_min(0), mean), torch.where(det, torch.zeros_like(var), var),
            torch.where(det, (m > 0).to(m.dtype), Phi))


class Prop:
    def __init__(self, W, G=40, var_dw=0.0, lookahead=(5, 9), nq=4096, chunk=6_000_000, grid="top", cellvar=False, exact_last=True, tiefix=True, tie_frac=1/3, k_la=3):
        self.wi, self.wh, self.wo = [torch.as_tensor(np.asarray(w, float), device=DEV, dtype=dt) for w in W]
        self.G = G
        if grid == "normal":
            e = norm.ppf(np.arange(G + 1) / G)
        else:   # cells of equal probability under 0.25 N(0,1) + 0.75 (average density of the three largest of ten)
            z = np.linspace(-6, 6, 200001); F = norm.cdf(z)
            from scipy.stats import binom
            Ftop = np.mean([binom.cdf(r - 1, 10, 1 - F) for r in (1, 2, 3)], 0)      # P(r-th largest <= z) = P(fewer than r above z)
            Fm = 0.25 * F + 0.75 * Ftop; e = np.interp(np.arange(G + 1) / G, Fm, z); e[0], e[-1] = -np.inf, np.inf
        pc = np.diff(norm.cdf(e)); mc = (norm.pdf(e[:-1]) - norm.pdf(e[1:])) / pc
        ex2 = 1 + (np.nan_to_num(e[:-1] * norm.pdf(e[:-1])) - np.nan_to_num(e[1:] * norm.pdf(e[1:]))) / pc
        self.edges = e; self.tiefix = tiefix
        wdt = np.diff(e); wdt = np.where(np.isfinite(wdt), wdt, np.sqrt(12 * np.maximum(ex2 - mc ** 2, 0)))
        self.dlt = torch.tensor(wdt * tie_frac, device=DEV, dtype=dt)   # a tie's typical spacing: a third of the cell width
        self.v = torch.tensor(mc, device=DEV, dtype=dt); self.pwv = torch.tensor(pc, device=DEV, dtype=dt)
        self.cv = torch.tensor(np.maximum(ex2 - mc ** 2, 0) * (1.0 if cellvar else 0.0), device=DEV, dtype=dt)
        if np.isscalar(var_dw): var_dw = (np.full(16, var_dw), np.full((16, 16), var_dw), np.full((10, 16), var_dw))
        self.vi, self.vh, self.vo = [torch.as_tensor(np.asarray(x, float), device=DEV, dtype=dt) for x in var_dw]
        self.noisy = bool((self.vi > 0).any() or (self.vh > 0).any() or (self.vo > 0).any())
        self.auto = lookahead == "auto"; self.collect = False; self.k_la = k_la
        self.la = [] if self.auto else list(lookahead); self.nb = 2 ** len(self.la); self.exact_last = exact_last
        Z = norm.ppf(np.clip(qmc.Sobol(9, scramble=True, seed=0).random(nq), 1e-12, 1 - 1e-12))
        self.Z = torch.tensor(Z, device=DEV, dtype=dt); self.chunk = chunk
        self.K = 10 * 11 * self.nb       # regime code = (pL * 11 + pS + 1) * nb + bits

    def _choose_lookahead(self, T, k=None):
        k = self.k_la if k is None else k
        """Automatic choice of the look-ahead neurons (no knowledge of roles): a first pass without look-ahead
        measures, per neuron, how much merging over the new number misplaces its next-step switch: the node's merged
        Gaussian gives P(on) for the next step, against the average of the per-value P(on) before merging.
        The k neurons with the largest total misplaced probability get look-ahead."""
        self.la = []; self.nb = 1; self.K = 110; self.collect = True; self.disc = torch.zeros(16, device=DEV, dtype=dt)
        ex = self.exact_last; self.exact_last = False
        self.run(T - 1); self.collect = False; self.exact_last = ex
        self.disc_all = self.disc.cpu().numpy()
        self.la = [int(j) for j in np.argsort(-self.disc_all)[:k]]; self.nb = 2 ** k; self.K = 110 * self.nb

    def _pk(self, iL, iS): return iL * (iL + 3) // 2 + iS

    def _unpk(self, q):
        iL = ((torch.sqrt(9 + 8 * q.to(dt)) - 3) / 2).floor().long()
        iL = torch.where(iL * (iL + 3) // 2 > q, iL - 1, iL); iL = torch.where((iL + 1) * (iL + 4) // 2 <= q, iL + 1, iL)
        return iL, q - iL * (iL + 3) // 2

    def _full(self, m2):
        E = torch.zeros(len(m2), 16, 16, device=DEV, dtype=dt); E[:, self.tri[0], self.tri[1]] = m2; E[:, self.tri[1], self.tri[0]] = m2
        return E

    def _bits(self, hm):
        b = torch.zeros(hm.shape[:-1], dtype=torch.long, device=DEV)
        for i, j in enumerate(self.la): b = b + ((hm @ self.wh[j]) > 0).long() * (2 ** i)
        return b

    def run(self, T=10):
        if self.auto and not self.collect and not self.la: self._choose_lookahead(T)
        G = self.G; K = self.K
        # packed storage: (L cell iL, S index iS <= iL + 1) -> iL (iL + 3) / 2 + iS; second moments as upper triangles
        self.NP = G * (G + 3) // 2; self.tri = torch.triu_indices(16, 16, device=DEV)
        self.W = torch.zeros(K, self.NP, device=DEV, dtype=dt); self.M1 = torch.zeros(K, self.NP, 16, device=DEV, dtype=dt)
        self.M2 = torch.zeros(K, self.NP, 136, device=DEV, dtype=dt)
        m = self.v[:, None] * self.wi[None]; vv = self.cv[:, None] * self.wi[None] ** 2
        h, hv, g = relu_moments(m, vv); code = (0 * 11 + 0) * self.nb + self._bits(h)
        C = g[:, :, None] * g[:, None, :] * self.cv[:, None, None] * self.wi[None, :, None] * self.wi[None, None, :]
        di = torch.arange(16, device=DEV); C[:, di, di] = hv
        ar = torch.arange(G, device=DEV); p = self.pwv
        pk = self._pk(ar, torch.zeros_like(ar)); E = p[:, None, None] * (C + h[:, :, None] * h[:, None, :])
        self.W[code, pk] = p; self.M1[code, pk] = p[:, None] * h; self.M2[code, pk] = E[:, self.tri[0], self.tri[1]]
        self.exact_last = getattr(self, "exact_last", True)
        for t in range(1, T): self.step(t, last=(t == T - 1 and self.exact_last))
        if self.exact_last: return self.acc_last
        return self.final()

    def step(self, t, last=False):
        G = self.G; nb = self.nb
        if last: self.acc_last = 0.0; self.accL_last = 0.0; self.mass = 0.0; self.by_pos = torch.zeros(10, 3, device=DEV, dtype=dt)
        Wo, M1o, M2o = self.W, self.M1, self.M2
        Wn = torch.zeros_like(Wo); M1n = torch.zeros_like(M1o); M2n = torch.zeros_like(M2o)
        if self.collect: Qn = torch.zeros_like(M1o)
        src = torch.nonzero(Wo > 0); n = len(src)
        per = max(1, self.chunk // (G * 256))
        di = torch.arange(16, device=DEV)
        for c0 in range(0, n, per):
            s = src[c0:c0 + per]; k, q = s[:, 0], s[:, 1]; iL, iS = self._unpk(q)
            w = Wo[k, q]; mu = M1o[k, q] / w[:, None]
            Sg = self._full(M2o[k, q]) / w[:, None, None] - mu[:, :, None] * mu[:, None, :]
            bits = k % nb; pS = (k // nb) % 11 - 1; pL = k // (nb * 11)
            ub = mu @ self.wh.T; Uc = self.wh @ Sg @ self.wh.T
            def moments(xv):
                m = ub[:, None, :] + xv[..., None] * self.wi[None, None, :]                       # [b, G, 16]
                vv = torch.diagonal(Uc, dim1=1, dim2=2)[:, None, :] + self.cv[None, :, None] * self.wi[None, None, :] ** 2
                if self.noisy:
                    vv = vv + (Eh2 @ self.vh.T)[:, None, :] + xv[..., None] ** 2 * self.vi[None, None, :]
                hm, hv, g = relu_moments(m, vv)
                for i, j in enumerate(self.la):
                    on = ((bits >> i) & 1).to(dt)[:, None]
                    hm[..., j] = on * m[..., j]; hv[..., j] = on * vv[..., j]; g[..., j] = on.expand_as(g[..., j])
                E2 = g[..., :, None] * g[..., None, :] * (Uc[:, None] + self.cv[None, :, None, None] * (self.wi[:, None] * self.wi[None, :])[None, None])
                E2[..., di, di] = hv
                return hm, E2 + hm[..., :, None] * hm[..., None, :], self._bits(hm)
            if self.noisy: Eh2 = torch.diagonal(Sg, dim1=1, dim2=2) + mu ** 2
            kx = torch.arange(G, device=DEV)[None, :]; Lb = iL[:, None]; Sb = iS[:, None]
            f_rec = (kx > Lb).to(dt) + 0.5 * (kx == Lb).to(dt); f_low = 1 - f_rec
            f_sec = f_low * ((kx + 1 > Sb).to(dt) + 0.5 * (kx + 1 == Sb).to(dt)); f_nei = f_low - f_sec
            ww = w[:, None] * self.pwv[None, :]
            B = len(s); kk = kx.expand(B, -1); Lx = Lb.expand(-1, G); Sx = Sb.expand(-1, G)
            pLx = pL[:, None].expand(-1, G); pSx = pS[:, None].expand(-1, G)
            self.npairs = getattr(self, "npairs", 0) + 3 * B * G                                   # Gaussian transitions computed
            tL = (kx == Lb).to(dt); tS = (kx + 1 == Sb).to(dt) * (1 - tL); d_ = self.dlt[None, :]
            xb = self.v[None, :].expand(B, -1)
            offs = (d_ * tL, d_ * (tS - tL), -d_ * tS) if self.tiefix else (0 * tL,) * 3
            for f, nL, nS, dL, dS, off in ((f_rec, torch.full_like(kk, t), pLx, kk, Lx + 1, offs[0]),
                                           (f_sec, pLx, torch.full_like(kk, t), Lx, kk + 1, offs[1]),
                                           (f_nei, pLx, pSx, Lx, Sx, offs[2])):
                hm, E2, nbits = moments(xb + off)
                wt = ww * f; sel = wt > 0
                if last:
                    sel = wt > 1e-13
                    if not sel.any(): continue
                    hs = hm[sel]; Cs = E2[sel] - hs[:, :, None] * hs[:, None, :]
                    pw_ = self.win(hs, Cs, nS[sel], nq=512); pl_ = self.win(hs, Cs, nL[sel], nq=512); ws = wt[sel]
                    self.nfinal = getattr(self, "nfinal", 0) + len(ws)
                    self.acc_last += (ws * pw_).sum().item(); self.accL_last += (ws * pl_).sum().item(); self.mass += ws.sum().item()
                    self.by_pos.index_add_(0, nS[sel], torch.stack([ws, ws * pw_, ws * pl_], 1))
                    continue
                code = ((nL * 11 + nS + 1) * nb + nbits)[sel]
                idx = code * self.NP + self._pk(dL[sel], dS[sel]); wsel = wt[sel]
                Wn.view(-1).index_add_(0, idx, wsel)
                if self.collect:
                    hs = hm[sel]; Cp = E2[sel] - hs[:, :, None] * hs[:, None, :]
                    dmn = hs @ self.wh.T; dv = torch.einsum("ja,nab,jb->nj", self.wh, Cp, self.wh) + self.wi ** 2
                    Qn.view(-1, 16).index_add_(0, idx, wsel[:, None] * torch.special.ndtr(dmn / dv.clamp_min(1e-30).sqrt()))
                M1n.view(-1, 16).index_add_(0, idx, wsel[:, None] * hm[sel])
                M2n.view(-1, 136).index_add_(0, idx, wsel[:, None] * E2[sel][:, self.tri[0], self.tri[1]])
        self.W, self.M1, self.M2 = Wn, M1n, M2n
        if self.collect:
            nz = torch.nonzero(Wn > 0); k_, q_ = nz[:, 0], nz[:, 1]; w_ = Wn[k_, q_]; mu_ = M1n[k_, q_] / w_[:, None]
            Sg_ = self._full(M2n[k_, q_]) / w_[:, None, None] - mu_[:, :, None] * mu_[:, None, :]
            dmn = mu_ @ self.wh.T; dv = torch.einsum("ja,nab,jb->nj", self.wh, Sg_, self.wh) + self.wi ** 2
            Pm = torch.special.ndtr(dmn / dv.clamp_min(1e-30).sqrt())
            self.disc += (w_[:, None] * Pm - Qn[k_, q_]).abs().sum(0)
        if last: self.share_L = self.accL_last / max(1e-12, self.mass - self.acc_last)

    def win(self, mu, Sg, k, nq=None):
        """P(score k is highest), per Gaussian state."""
        Zq = self.Z if nq is None else self.Z[:nq]
        sm = mu @ self.wo.T; sS = self.wo @ Sg @ self.wo.T
        if self.noisy: sS = sS + torch.diag_embed((torch.diagonal(Sg, dim1=1, dim2=2) + mu ** 2) @ self.vo.T)
        n = len(mu)
        oth = torch.stack([torch.cat([torch.arange(j, device=DEV), torch.arange(j + 1, 10, device=DEV)]) for j in range(10)])[k]
        A = torch.zeros(n, 9, 10, device=DEV, dtype=dt); A[torch.arange(n, device=DEV)[:, None], torch.arange(9, device=DEV)[None], k[:, None]] = 1
        A.scatter_(2, oth[:, :, None], -1.0)
        dm = (A @ sm[:, :, None])[..., 0]; dS = A @ sS @ A.transpose(1, 2)
        ev, V = torch.linalg.eigh(dS); Lc = V * ev.clamp_min(1e-12).sqrt()[:, None, :]
        out = []
        step_ = max(1, 4_000_000 // len(Zq))
        for c0 in range(0, n, step_):
            sl = slice(c0, c0 + step_)
            out.append(((dm[sl, None, :] + torch.einsum("qj,nij->nqi", Zq, Lc[sl])) > 0).all(2).to(dt).mean(1))
        return torch.cat(out)

    def final(self):
        nz = torch.nonzero(self.W > 0); k, q = nz[:, 0], nz[:, 1]; iL, iS = self._unpk(q)
        w = self.W[k, q]; mu = self.M1[k, q] / w[:, None]
        Sg = self._full(self.M2[k, q]) / w[:, None, None] - mu[:, :, None] * mu[:, None, :]; Sg = 0.5 * (Sg + Sg.transpose(1, 2))
        pS = (k // self.nb) % 11 - 1; pL = k // (self.nb * 11)
        self.node = dict(w=w, pL=pL, pS=pS, iL=iL, iS=iS, k=k)
        self.p_right = self.win(mu, Sg, pS); self.p_L = self.win(mu, Sg, pL)
        self.mass = w.sum().item(); acc = (w * self.p_right).sum().item()
        self.share_L = (w * self.p_L).sum().item() / max(1e-12, self.mass - acc)
        return acc
