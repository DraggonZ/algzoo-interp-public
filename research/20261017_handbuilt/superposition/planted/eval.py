"""Evaluate the planted SAEs.
(1) Held-out unexplained variance: from train.log (repeated here on fresh lists).
(2) Role recovery, on 60,000 fresh random lists, all steps pooled, after removing from every role and every latent the
    part explained linearly by the largest so far, the current number and the step (as ../sae/eval_roles_fair.py):
    Shared set = largest so far before and after the current number, the number, the step (see common_planted.shared).
    Store and detect are exact functions of the shared set, so they enter only the 'raw' count.
    - per neuron: best |correlation| with any single latent; count above 0.9 (all neurons; and 'single' neurons, i.e.
      not in a rotating group or the memory). 'raw' = the same without removing the shared part.
    - per subspace (rotating groups; memory as one subspace): pick as many latents as the group has neurons, greedily,
      to maximise the summed squared canonical correlations with the group; report the smallest canonical
      correlation (1 = the group's whole subspace is spanned). 'all lat' = the same using every live latent (a linear
      read-out ceiling).
    Reference dictionaries: the network's own neurons (1 by construction), a random rotation of the scaled state, and
    its principal components; for the projection, the 16 projected dimensions themselves.
(3) Splicing (v21, v19): state replaced by the SAE reconstruction at the last step only or at every step; accuracy on
    200,000 random lists (naming the second-largest). Control: Gaussian noise with the same per-neuron size as the
    SAE's reconstruction error, added at the same steps."""
import sys
from common_planted import *
def load(setup, mult, k):
    D = torch.load(f"saes_{setup}.pt"); d = 16 if setup != "v21" else 28; s = SAE(mult * d, k, d)
    s.load_state_dict({kk: v.to(DEV) for kk, v in D[(mult, k)].items()}); return s, {kk: v.to(DEV) for kk, v in D.items() if isinstance(kk, str)}
def encode_in(setup, H, norm):
    Hn = H / norm["sd"].to(dt)
    if setup == "v21proj": Hn = (Hn @ PROJ.T) / norm["sdp"].to(dt)
    return Hn
def per_neuron(R, Z, keepR):
    a = (R - R.mean(0)) / (R.std(0) + 1e-12); b = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return (a.T @ b / len(a)).abs().max(1).values[keepR]
def cancorr(Crr, Crz, Czz, idx):
    e, V = torch.linalg.eigh(Crr); Ri = V @ torch.diag(e.clamp_min(1e-12 * e.max()) ** -0.5) @ V.T
    B = Crz[:, idx]; Ci = torch.linalg.pinv(Czz[np.ix_(idx, idx)], hermitian=True, rtol=1e-9)
    M = Ri @ B @ Ci @ B.T @ Ri; return torch.linalg.eigvalsh(M).clamp(0, 1).sqrt()
def group_recovery(R, Z, g):
    Rg = R[:, g]; Rg = (Rg - Rg.mean(0)) / Rg.std(0); Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12); n = len(R)   # correlation scale
    Crr = Rg.T @ Rg / n; Crz = Rg.T @ Zc / n; Czz = Zc.T @ Zc / n; chosen = []
    for _ in range(len(g)):
        best = max((c for c in range(Z.shape[1]) if c not in chosen), key=lambda c: (cancorr(Crr, Crz, Czz, chosen + [c]) ** 2).sum().item())
        chosen.append(best)
    return cancorr(Crr, Crz, Czz, chosen).min().item(), cancorr(Crr, Crz, Czz, list(range(Z.shape[1]))).min().item()
def splice_acc(W, X, f, every):
    y = X.argsort(1, descending=True)[:, 1]; h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if every or t == 9: h = f(h)
    return 100 * ((h @ W[2].T).argmax(1) == y).double().mean().item()
if __name__ == "__main__":
    setups = sys.argv[1:] or ["v21", "v19", "v21proj"]
    g = torch.Generator(device=DEV).manual_seed(78); X = torch.randn(60000, 10, device=DEV, dtype=dt, generator=g); F = shared(X)
    g2 = torch.Generator(device=DEV).manual_seed(79); Xs = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g2)
    for setup in setups:
        net = "v19" if setup == "v19" else "v21"; W, names = NETS[net]; d = len(W[0])
        H = states(W, X).reshape(-1, d); Rr = resid(H, F); keepR = Rr.std(0) > 1e-9 * H.std(0)     # roles not fully explained by the shared quantities (store and detect are)
        single = [i for i in range(d) if i not in MEM[net] and not any(i in gg for gg in ROT[net].values())]
        groups = dict(ROT[net]); groups["memory"] = MEM[net]
        print(f"\n=== {setup}: roles = {net}'s {d} neurons; {int(keepR.sum())} with a non-zero part after removing shared quantities; "
              f"{len(single)} single neurons ({', '.join(names[i] for i in single)}); subspaces: " + ", ".join(f"{k} ({len(v)})" for k, v in groups.items()), flush=True)
        _, norm0 = load(setup, 1, 2); Hn = encode_in(setup, H, norm0)
        dicts = {}
        if setup == "v21proj": dicts["16 projected dims"] = (None, Hn)
        else:
            dicts["own neurons"] = (None, H)
            Q = torch.linalg.qr(torch.tensor(np.random.default_rng(5).standard_normal((d, d)), device=DEV, dtype=dt))[0]
            dicts["random rotation"] = (None, Hn @ Q)
            Hc = Hn - Hn.mean(0); dicts["principal components"] = (None, Hc @ torch.linalg.eigh(Hc.T @ Hc)[1])
        for mult in (1, 2, 4):
            for k in (2, 4, 8):
                s, norm = load(setup, mult, k)
                with torch.no_grad():
                    r, z = s(Hn.float()); fvu = 100 * (((r.to(dt) - Hn) ** 2).sum() / ((Hn - Hn.mean(0)) ** 2).sum()).item()
                dicts[f"SAE {mult}x k{k}"] = ((s, norm, fvu), z.to(dt))
        print(f"{'dictionary':21s} | unexpl. | per neuron: mean best, >0.9 all, >0.9 single, raw >0.9 | subspaces, smallest canon. corr. with {{own size}} latents (all latents) | "
              + ("spliced last / every step; noise-matched last / every" if setup != "v21proj" else ""), flush=True)
        if setup == "v21proj":
            A = Hn - Hn.mean(0); B = Rr - Rr.mean(0)          # linear read-out of each role from the 16 dims (no shared removal on the dims: fair to the dims)
            Ar = resid(Hn, F); Ar = Ar - Ar.mean(0); pred = Ar @ torch.linalg.lstsq(Ar, B).solution
            mc = ((pred * B).sum(0) / (pred.norm(dim=0) * B.norm(dim=0) + 1e-12))[keepR]
            print(f"{'(linear from 16 dims)':21s} |         | multiple corr. mean {mc.mean():.2f}, >0.9 {(mc > 0.9).sum().item()}/{len(mc)}", flush=True)
        for nm, (meta, Z) in dicts.items():
            live = Z.std(0) > 1e-9; Z = Z[:, live]; Zr = resid(Z, F)
            bm = per_neuron(Rr, Zr, keepR); braw = per_neuron(H, Z, H.std(0) > 0)
            ks = torch.tensor([i in single for i in range(d)], device=DEV)[keepR]
            gr = {gn: group_recovery(Rr, Zr, gi) for gn, gi in groups.items()}
            line = (f"{nm:21s} | " + (f"{meta[2]:6.2f}% | " if meta else "        | ") +
                    f"{bm.mean():.2f}, {(bm > 0.9).sum().item():2d}/{len(bm)}, {(bm[ks] > 0.9).sum().item()}/{int(ks.sum())}, {(braw > 0.9).sum().item():2d}/{len(braw)} | " +
                    ", ".join(f"{gn} {a:.2f} ({b:.2f})" for gn, (a, b) in gr.items()))
            if meta and setup != "v21proj":
                s, norm, _ = meta; sd = norm["sd"].to(dt)
                f = lambda h: (s((h / sd).float())[0].to(dt) * sd)
                with torch.no_grad():
                    err = (s(Hn.float())[0].to(dt) * sd - H).pow(2).mean(0).sqrt()       # per-neuron RMS reconstruction error
                    nz = lambda h: h + err * torch.randn_like(h)
                    line += f" | {splice_acc(W, Xs, f, False):5.1f} / {splice_acc(W, Xs, f, True):5.1f}; {splice_acc(W, Xs, nz, False):5.1f} / {splice_acc(W, Xs, nz, True):5.1f}"
            print(line + f"  [{int(live.sum())} live]", flush=True)
            if nm in ("SAE 4x k8", "SAE 1x k4"):
                print("   per neuron best (shared removed): " + " ".join(f"{names[i]}={v:.2f}" for i, v in zip(torch.nonzero(keepR).flatten().tolist(), bm.tolist())), flush=True)
