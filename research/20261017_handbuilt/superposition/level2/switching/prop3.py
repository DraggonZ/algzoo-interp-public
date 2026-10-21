"""Forward propagation of a probabilistic model of the hidden state (Hilton's proposal), version 3: a conditional
Gaussian model with the list quantities kept exact.

Why: version 2 (prop2.py) put the largest and second-largest so far into the same Gaussian as the state. The order
constraint 'x > L' then cannot be represented: the moment-matched piece straddles x = L, and neurons that read
x - L with large weights (hiding neuron 7, the late gate at -66 per unit) get huge spurious spread. Result 24%.

Representation. Regime = (position of the largest so far, position of the second-largest so far). Within a regime the
values L and S sit on a fixed grid of G values (midpoints of G equal-probability cells of N(0,1)); the new number x
uses the same grid, so every list event (new largest, new second, neither) is exact and lands on a grid node. At
each (regime, L node, S node) the state h is one Gaussian (mean, covariance). Per step, for each source node and
each x node: the pre-activation u = W_hh h + w_in x is Gaussian; each neuron's ReLU is applied with exact mean and
variance; cross-covariances scale by the gains P(u_j > 0) (approximate between two ReLUs). The result is added
(moment matching) to the destination node. Ties (x in the same cell as L or S) split the weight half and half.
End: P(the second-largest's score is the highest) per node from its Gaussian score distribution (fixed quasi-random
points), weighted.

Gap as noise: var_dw = per-weight variances of (W_actual - W_used); each neuron's input gets independent zero-mean
noise of variance sum_k var(dW_hh[j,k]) E[h_k^2] + var(dw_in[j]) x^2 (fresh each step), and the scores
sum_k var(dW_out[i,k]) E[h_k^2]."""
import numpy as np, torch
from scipy.stats import norm, qmc

DEV = "cuda"; dt = torch.float64


def _relu_moments(m, v):
    """Elementwise ReLU of N(m, v): mean, variance, gain P(u > 0)."""
    s = v.clamp_min(1e-30).sqrt(); r = m / s
    Phi = torch.special.ndtr(r); phi = torch.exp(-0.5 * r * r) / np.sqrt(2 * np.pi)
    mean = m * Phi + s * phi; var = ((m * m + v) * Phi + m * s * phi - mean * mean).clamp_min(0)
    det = v < 1e-20
    pos = (m > 0).to(m.dtype)
    return torch.where(det, m.clamp_min(0), mean), torch.where(det, torch.zeros_like(var), var), torch.where(det, pos, Phi)


class Prop:
    def __init__(self, W, G=60, var_dw=0.0, nq=4096, lookahead=(), verbose=False):
        self.wi, self.wh, self.wo = [torch.as_tensor(np.asarray(w, float), device=DEV, dtype=dt) for w in W]
        self.G = G; u = (np.arange(G) + 0.5) / G
        self.v = torch.tensor(norm.ppf(u), device=DEV, dtype=dt); self.pw = 1.0 / G
        if np.isscalar(var_dw):
            var_dw = (np.full(16, var_dw), np.full((16, 16), var_dw), np.full((10, 16), var_dw))
        self.vi, self.vh, self.vo = [torch.as_tensor(np.asarray(x, float), device=DEV, dtype=dt) for x in var_dw]
        self.noisy = bool((self.vi > 0).any() or (self.vh > 0).any() or (self.vo > 0).any())
        Z = norm.ppf(np.clip(qmc.Sobol(9, scramble=True, seed=0).random(nq), 1e-12, 1 - 1e-12))
        self.Z = torch.tensor(Z, device=DEV, dtype=dt); self.verbose = verbose
        self.la = list(lookahead)   # neurons switched by the sign of their drive decided one step earlier (the gates' rule)

    # state per regime: (W [G, G+1], M1 [G, G+1, 16], M2 [G, G+1, 16, 16]); S index 0 = none, j+1 = node j
    def _new(self):
        G = self.G
        return [torch.zeros(G, G + 1, device=DEV, dtype=dt), torch.zeros(G, G + 1, 16, device=DEV, dtype=dt),
                torch.zeros(G, G + 1, 16, 16, device=DEV, dtype=dt)]

    def run(self, T=10):
        G = self.G; st = {}
        # step 0: x is the largest; h = relu(w_in x), exact
        h = torch.relu(self.v[:, None] * self.wi[None])
        bits = self._bits(h)
        for b in torch.unique(bits).tolist():
            r = self._new(); m = (bits == b).to(dt)
            r[0][:, 0] = self.pw * m; r[1][:, 0] = self.pw * m[:, None] * h; r[2][:, 0] = self.pw * m[:, None, None] * h[:, :, None] * h[:, None, :]
            st[(0, -1, b)] = r
        for t in range(1, T):
            st = self.step(st, t)
            if self.verbose: print(f"step {t}: {len(st)} regimes, mass {sum(r[0].sum().item() for r in st.values()):.6f}", flush=True)
        self.st = st
        return self.final(st)

    def _bits(self, hm):
        b = torch.zeros(hm.shape[:-1], dtype=torch.long, device=DEV)
        for i, j in enumerate(self.la): b = b + ((hm @ self.wh[j]) > 0).long() * (2 ** i)
        return b

    def step(self, st, t):
        G = self.G; out = {}
        kx = torch.arange(G, device=DEV)
        for (pL, pS, bb), (Wn, M1, M2) in st.items():
            nz = Wn > 0
            iL, iS = torch.nonzero(nz, as_tuple=True); w = Wn[nz]
            mu = M1[nz] / w[:, None]; Sg = M2[nz] / w[:, None, None] - mu[:, :, None] * mu[:, None, :]
            ub = mu @ self.wh.T; Uc = self.wh @ Sg @ self.wh.T                       # [n,16], [n,16,16]
            if self.noisy:
                Eh2 = torch.diagonal(Sg, dim1=1, dim2=2) + mu ** 2
                qh = Eh2 @ self.vh.T                                                    # [n,16]
            n = len(w)
            for c0 in range(0, n, max(1, 200000 // G)):
                sl = slice(c0, min(n, c0 + max(1, 200000 // G)))
                m = ub[sl, None, :] + self.v[None, :, None] * self.wi[None, None, :]     # [b, G, 16]
                Ud = torch.diagonal(Uc[sl], dim1=1, dim2=2)[:, None, :].expand_as(m)
                vv = Ud.clone()
                if self.noisy: vv = vv + qh[sl, None, :] + self.v[None, :, None] ** 2 * self.vi[None, None, :]
                hm, hv, g = _relu_moments(m, vv)
                for i, j in enumerate(self.la):                                         # forced by last step's rule
                    on = float((bb >> i) & 1)
                    hm[..., j] = on * m[..., j]; hv[..., j] = on * vv[..., j]; g[..., j] = on
                nb = self._bits(hm)                                                     # [b, G] next step's gate states
                C = g[..., :, None] * g[..., None, :] * Uc[sl, None]                    # [b, G, 16, 16]
                di = torch.arange(16, device=DEV); C[..., di, di] = hv
                E2 = C + hm[..., :, None] * hm[..., None, :]
                ww = w[sl, None] * self.pw                                              # [b, 1] -> per x
                Lb, Sb = iL[sl][:, None], iS[sl][:, None]                               # S index (0 = none)
                k = kx[None, :]
                # event weights: record, second, neither (ties split half and half)
                f_rec = (k > Lb).to(dt) + 0.5 * (k == Lb).to(dt)
                f_low = 1 - f_rec
                kS = k + 1
                f_sec = f_low * ((kS > Sb).to(dt) + 0.5 * (kS == Sb).to(dt))
                f_nei = f_low - f_sec
                Lx = Lb.expand(-1, G); Sx = Sb.expand(-1, G); kk = k.expand(len(Lb), -1)
                for key0, f, dL, dS in (((t, pL), f_rec, kk, Lx + 1), ((pL, t), f_sec, Lx, kk + 1), ((pL, pS), f_nei, Lx, Sx)):
                  for b2 in range(2 ** len(self.la)):
                    key = key0 + (b2,)
                    wt = ww * f * (nb == b2).to(dt); sel = wt > 0
                    if not sel.any(): continue
                    if key not in out: out[key] = self._new()
                    o = out[key]; idx = dL[sel] * (G + 1) + dS[sel]; wsel = wt[sel]
                    o[0].view(-1).index_add_(0, idx, wsel)
                    o[1].view(-1, 16).index_add_(0, idx, wsel[:, None] * hm[sel])
                    o[2].view(-1, 16, 16).index_add_(0, idx, wsel[:, None, None] * E2[sel])
        return out

    def _win(self, mu, Sg, k):
        """P(score k is the highest) per Gaussian; mu [n,16], Sg [n,16,16], k [n] or int."""
        sm = mu @ self.wo.T; sS = self.wo @ Sg @ self.wo.T
        if self.noisy:
            Eh2 = torch.diagonal(Sg, dim1=1, dim2=2) + mu ** 2
            sS = sS + torch.diag_embed(Eh2 @ self.vo.T)
        n = len(mu); kk = torch.as_tensor(k, device=DEV).expand(n)
        A = torch.zeros(n, 9, 10, device=DEV, dtype=dt)
        oth = torch.stack([torch.cat([torch.arange(j, device=DEV), torch.arange(j + 1, 10, device=DEV)]) for j in range(10)])[kk]
        r9 = torch.arange(9, device=DEV)
        A[:, r9, kk[:, None].expand(-1, 9)] = 1; A.scatter_(2, oth[:, :, None], -1.0)
        dm = (A @ sm[:, :, None])[..., 0]; dS = A @ sS @ A.transpose(1, 2) + 1e-9 * torch.eye(9, device=DEV, dtype=dt)
        ev, V = torch.linalg.eigh(dS); Lc = V * ev.clamp_min(1e-9).sqrt()[:, None, :]
        out = []
        for c0 in range(0, n, 2000):
            sl = slice(c0, c0 + 2000)
            smp = dm[sl, None, :] + torch.einsum("qj,nij->nqi", self.Z, Lc[sl])
            out.append((smp > 0).all(2).to(dt).mean(1))
        return torch.cat(out)

    def final(self, st):
        acc = 0.0; mL = 0.0; self.per = {}
        for (pL, pS, bb), (Wn, M1, M2) in st.items():
            nz = Wn > 0; w = Wn[nz]
            mu = M1[nz] / w[:, None]; Sg = M2[nz] / w[:, None, None] - mu[:, :, None] * mu[:, None, :]
            Sg = 0.5 * (Sg + Sg.transpose(1, 2))
            pw = self._win(mu, Sg, pS); pl = self._win(mu, Sg, pL)
            a = (w * pw).sum().item(); b = (w * pl).sum().item()
            self.per[(pL, pS, bb)] = (w.sum().item(), a, b); acc += a; mL += b
        self.mass = sum(v[0] for v in self.per.values())
        self.share_L = mL / max(1e-12, self.mass - acc)
        return acc
