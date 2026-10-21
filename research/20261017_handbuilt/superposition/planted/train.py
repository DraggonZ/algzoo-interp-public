"""Train TopK SAEs (recipe of ../sae/train_sae.py: Adam 6,000 steps of 8,192 states, lr 1e-3 then 3e-4 after 4,000,
no dead-latent revival) on each planted setup, widths 1x, 2x, 4x the input dimension, k 2, 4, 8. States from 400,000
random lists, all ten steps pooled; last 500,000 states held out for unexplained variance. Saves saes_<setup>.pt."""
import sys
from common_planted import *
def data(setup, X):
    W, _ = NETS["v19" if setup == "v19" else "v21"]
    H = states(W, X).reshape(-1, len(W[0]))
    return H
if __name__ == "__main__":
    setups = sys.argv[1:] or ["v21", "v19", "v21proj"]
    g = torch.Generator(device=DEV).manual_seed(401); X = torch.randn(400000, 10, device=DEV, dtype=dt, generator=g)
    for setup in setups:
        H = data(setup, X); sd = H.std(0) + 1e-6; Hn = H / sd; out = {"sd": sd.cpu()}
        if setup == "v21proj":
            Hn = Hn @ PROJ.T; sdp = Hn.std(0); Hn = Hn / sdp; out["sdp"] = sdp.cpu()
        Hn = Hn.to(F32); d = Hn.shape[1]; tr, te = Hn[:-500000], Hn[-500000:]
        for mult in (1, 2, 4):
            m = mult * d
            for k in (2, 4, 8):
                torch.manual_seed(m * 10 + k); s = SAE(m, k, d); s.bd.data = tr[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
                for step in range(6000):
                    b = tr[torch.randint(0, len(tr), (8192,), device=DEV)]; r, z = s(b); loss = ((r - b) ** 2).sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    if step == 4000:
                        for pg in opt.param_groups: pg["lr"] = 3e-4
                with torch.no_grad():
                    r, z = s(te); fvu = ((r - te) ** 2).sum() / ((te - te.mean(0)) ** 2).sum(); dead = ((z > 0).sum(0) == 0).sum().item()
                print(f"{setup:8s} width {m:3d} ({mult}x) k {k}: unexplained variance {100*fvu:.2f}% (held out), dead latents {dead}", flush=True)
                out[(mult, k)] = {kk: v.detach().cpu() for kk, v in s.state_dict().items()}
        torch.save(out, f"saes_{setup}.pt")
