"""TopK transcoders for ARC's recurrent cell. Input u = (h_{t-1} / sd, x_t) (17 numbers, h_0 = 0), target h_t / sd
(16 numbers), sd = per-neuron std of ARC's states over all steps (as in ../sae/train_sae.py). Encoder TopK(ReLU(W_e u + b_e)),
decoder W_d z + b_d with unit-norm columns (linear output; no ReLU, so splicing can feed back small negatives, see
eval_tc.py). Adam, 6,000 steps of 8,192 pairs, lr 1e-3 then 3e-4 after step 4,000, no dead-latent revival. Widths 16, 32,
64, 128; k 2, 4, 8. Data: 800,000 random lists, all ten steps (8 M pairs), last 1 M held out. Saved to tcs.pt.
Reports: held-out unexplained variance of the next state, dead latents."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
F32 = torch.float32
def states(X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt); H = []
    for t in range(10): h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0]); H.append(h)
    return torch.stack(H, 1)
def pairs(X, sd):
    """(inputs, targets) per list and step: u_t = (h_{t-1}/sd, x_t), y_t = h_t/sd; shapes (n, 10, 17), (n, 10, 16)."""
    H = states(X).to(F32) / sd; Hp = torch.cat([torch.zeros_like(H[:, :1]), H[:, :-1]], 1)
    return torch.cat([Hp, X.to(F32)[..., None]], 2), H
class TC(torch.nn.Module):
    def __init__(self, m, k, din=17, dout=16):
        super().__init__(); self.k = k
        self.We = torch.nn.Parameter(torch.randn(m, din, device=DEV) / np.sqrt(din)); self.be = torch.nn.Parameter(torch.zeros(m, device=DEV))
        self.Wd = torch.nn.Parameter(torch.randn(dout, m, device=DEV) / np.sqrt(m)); self.bd = torch.nn.Parameter(torch.zeros(dout, device=DEV))
    def encode(self, u):
        a = torch.relu(u @ self.We.T + self.be); v, i = a.topk(self.k, -1)
        return torch.zeros_like(a).scatter(-1, i, v)
    def forward(self, u):
        z = self.encode(u); Wd = self.Wd / self.Wd.norm(dim=0, keepdim=True); return z @ Wd.T + self.bd, z
def fvu(r, y): return (((r - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()).item()
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(301)
    X = torch.randn(800000, 10, device=DEV, dtype=dt, generator=g)
    sd = states(X).reshape(-1, 16).std(0).to(F32) + 1e-6
    U, Y = pairs(X, sd); U, Y = U.reshape(-1, 17), Y.reshape(-1, 16)
    tr, te = slice(0, 7000000), slice(7000000, None); Ute, Yte = U[te][:500000], Y[te][:500000]
    out = {"sd": sd.cpu()}
    for m in (16, 32, 64, 128):
        for k in (2, 4, 8):
            torch.manual_seed(m * 10 + k); s = TC(m, k); s.bd.data = Y[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
            for step in range(6000):
                ix = torch.randint(0, 7000000, (8192,), device=DEV); r, z = s(U[ix]); loss = ((r - Y[ix]) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if step == 4000:
                    for pg in opt.param_groups: pg["lr"] = 3e-4
            with torch.no_grad():
                r, z = s(Ute); dead = ((z > 0).sum(0) == 0).sum().item()
            print(f"width {m:3d} k {k}: unexplained variance of next state {100*fvu(r, Yte):.2f}% (held out), dead latents {dead}", flush=True)
            out[(m, k)] = {kk: v.detach().cpu() for kk, v in s.state_dict().items()}
    torch.save(out, "tcs.pt")
