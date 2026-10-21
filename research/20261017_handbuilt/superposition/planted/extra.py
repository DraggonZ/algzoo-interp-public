"""Two controls for the planted test.
(a) How sparse are the planted roles? Number of active neurons per state (v21, v19). A TopK SAE with k below that
    count cannot use the neuron basis as its dictionary.
(b) Does a larger k let the SAE find the neurons? 1x-wide SAEs with k = 12, 16, 20 (v21) and 10, 12 (v19), same recipe;
    per-neuron best |correlation| (shared part removed, as eval.py) and raw.
(c) Recipe sanity check on genuinely sparse planted features: 28 random unit directions in 16 dimensions, each active
    with probability 0.08 with an exponential magnitude; SAE 2x wide, k 4 (same recipe). Best |cosine| between each
    true direction and any decoder column, and best |correlation| of each true activation with any latent."""
from common_planted import *
from eval import per_neuron
def train(Hn, m, k, seed):
    torch.manual_seed(seed); d = Hn.shape[1]; s = SAE(m, k, d); tr = Hn[:-200000]; s.bd.data = tr[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
    for step in range(6000):
        b = tr[torch.randint(0, len(tr), (8192,), device=DEV)]; r, z = s(b); loss = ((r - b) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step == 4000:
            for pg in opt.param_groups: pg["lr"] = 3e-4
    te = Hn[-200000:]
    with torch.no_grad(): r, z = s(te); fvu = 100 * (((r - te) ** 2).sum() / ((te - te.mean(0)) ** 2).sum()).item()
    return s, fvu
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(401); X = torch.randn(400000, 10, device=DEV, dtype=dt, generator=g)
    g2 = torch.Generator(device=DEV).manual_seed(78); Xe = torch.randn(60000, 10, device=DEV, dtype=dt, generator=g2); F = shared(Xe)
    for net, ks in (("v21", (12, 16, 20)), ("v19", (10, 12))):
        W, names = NETS[net]; d = len(W[0]); H = states(W, X).reshape(-1, d); act = (H > 0).sum(1).double()
        q = torch.quantile(act[:1000000], torch.tensor([0.5, 0.9, 0.99, 1.0], device=DEV, dtype=dt))
        print(f"{net}: active neurons per state mean {act.mean():.1f}, median / 90% / 99% / max {q.tolist()}; at steps 3-9 mean {(H.reshape(-1, 10, d)[:, 3:] > 0).sum(2).double().mean():.1f}", flush=True)
        sd = H.std(0) + 1e-6; Hn = (H / sd).float(); He = states(W, Xe).reshape(-1, d); Rr = resid(He, F); keepR = Rr.std(0) > 1e-9 * He.std(0)
        for k in ks:
            s, fvu = train(Hn, d, k, 1000 + k)
            with torch.no_grad(): z = s.encode((He / sd).float()).to(dt)
            z = z[:, z.std(0) > 1e-9]; bm = per_neuron(Rr, resid(z, F), keepR); braw = per_neuron(He, z, He.std(0) > 0)
            print(f"  1x k {k}: unexplained {fvu:.2f}%, per neuron (shared removed) mean {bm.mean():.2f}, >0.9 {(bm > 0.9).sum().item()}/{len(bm)}; raw >0.9 {(braw > 0.9).sum().item()}/{len(braw)}", flush=True)
    rng = np.random.default_rng(7); Dt = torch.tensor(rng.standard_normal((28, 16)), device=DEV, dtype=F32); Dt = Dt / Dt.norm(dim=1, keepdim=True)
    A = (torch.rand(2000000, 28, device=DEV) < 0.08).float() * torch.empty(2000000, 28, device=DEV).exponential_()
    Y = A @ Dt; sdY = Y.std(0); Yn = Y / sdY
    for k in (2, 4):
        s, fvu = train(Yn, 32, k, 7 + k)
        Wd = (s.Wd / s.Wd.norm(dim=0, keepdim=True)).detach() * sdY[:, None]; Wd = Wd / Wd.norm(dim=0, keepdim=True)
        cos = (Dt @ Wd).abs().max(1).values
        with torch.no_grad(): z = s.encode(Yn[-200000:])
        a = A[-200000:]; a = (a - a.mean(0)) / a.std(0); zz = z[:, z.std(0) > 0]; zz = (zz - zz.mean(0)) / zz.std(0); c = (a.T @ zz / len(a)).abs().max(1).values
        print(f"toy (28 sparse features in 16 dims, 8% active each), SAE 2x k {k}: unexplained {fvu:.2f}%, decoder |cos| mean {cos.mean():.2f}, >0.9 {(cos > 0.9).sum().item()}/28; activation corr mean {c.mean():.2f}, >0.9 {(c > 0.9).sum().item()}/28", flush=True)
