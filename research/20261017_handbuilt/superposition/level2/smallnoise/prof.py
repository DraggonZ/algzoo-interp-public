import time, torch, models
from lib import *
W = load_net("ARC"); n = 4000
pos = TRIPLES[torch.randint(0, 720, (n,), device=DEV)]; val = torch.sort(torch.randn(n, 3, device=DEV, dtype=dt).abs(), 1, descending=True).values
def T(f, *a):
    torch.cuda.synchronize(); t0 = time.time(); r = f(*a); torch.cuda.synchronize(); return r, time.time() - t0
X = torch.randn(n * 28, 10, device=DEV, dtype=dt)
_, t = T(run_jac, W, X); print("run_jac", t)
lm = torch.randn(n * 28, 3, device=DEV, dtype=dt); lc = torch.eye(3, device=DEV, dtype=dt).expand(n * 28, 3, 3).contiguous()
models.gauss_split(lm, lc); _, t = T(models.gauss_split, lm, lc); print("gauss_split", t)
_, t = T(models.model_fourth, W, pos, val, 4); print("model_fourth total", t)
