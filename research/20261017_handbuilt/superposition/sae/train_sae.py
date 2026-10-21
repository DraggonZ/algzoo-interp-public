"""TopK sparse autoencoders on ARC's hidden states (all 16 neurons, all ten steps, random lists). Encoder: TopK(ReLU(W_e (h - b_d) + b_e)),
decoder: W_d z + b_d with unit-norm columns. Adam, 6,000 steps of 8,192 states; dead latents revived by the auxiliary
loss of Gao et al. (2024) is skipped for simplicity (dead counts are reported). Widths 16, 32, 64, 128; k 2, 4, 8.
Saved to saes.pt. Reports: fraction of variance unexplained (held out), dead latents."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
F32 = torch.float32
def states(X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt); H = []
    for t in range(10): h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0]); H.append(h)
    return torch.stack(H, 1)
class SAE(torch.nn.Module):
    def __init__(self, m, k, d=16):
        super().__init__(); self.k = k
        self.We = torch.nn.Parameter(torch.randn(m, d, device=DEV) / np.sqrt(d)); self.be = torch.nn.Parameter(torch.zeros(m, device=DEV))
        self.Wd = torch.nn.Parameter(self.We.data.T.clone()); self.bd = torch.nn.Parameter(torch.zeros(d, device=DEV))
    def encode(self, h):
        a = torch.relu((h - self.bd) @ self.We.T + self.be); v, i = a.topk(self.k, -1)
        return torch.zeros_like(a).scatter(-1, i, v)
    def forward(self, h):
        z = self.encode(h); Wd = self.Wd / self.Wd.norm(dim=0, keepdim=True); return z @ Wd.T + self.bd, z
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(301)
    H = states(torch.randn(800000, 10, device=DEV, dtype=dt, generator=g)).reshape(-1, 16).to(F32)
    sd = H.std(0) + 1e-6; Hn = H / sd                     # per-neuron scaling (undone when splicing)
    tr, te = Hn[:7000000], Hn[7000000:]
    out = {"sd": sd.cpu()}
    for m in (16, 32, 64, 128):
        for k in (2, 4, 8):
            torch.manual_seed(m * 10 + k); s = SAE(m, k); s.bd.data = tr[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
            for step in range(6000):
                b = tr[torch.randint(0, len(tr), (8192,), device=DEV)]; r, z = s(b); loss = ((r - b) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if step == 4000:
                    for pg in opt.param_groups: pg["lr"] = 3e-4
            with torch.no_grad():
                r, z = s(te[:500000]); fvu = ((r - te[:500000]) ** 2).sum() / ((te[:500000] - te[:500000].mean(0)) ** 2).sum()
                dead = ((z > 0).sum(0) == 0).sum().item()
            print(f"width {m:3d} k {k}: unexplained variance {100*fvu:.2f}% (held out), dead latents {dead}", flush=True)
            out[(m, k)] = {kk: v.detach().cpu() for kk, v in s.state_dict().items()}
    torch.save(out, "saes.pt")
