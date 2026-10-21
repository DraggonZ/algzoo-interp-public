"""Cheap faithfulness fix: fine-tune trained transcoders with ARC's cell replaced at every step (own states fed back,
ten steps unrolled), loss = mean squared next-state error along the spliced run (scaled states, vs ARC's states on the
same list) + lam * KL(softmax ARC scores || softmax spliced scores). Adam 3e-4, 3,000 steps of 4,096 fresh random lists.
Reference: ARC with Gaussian noise added to its states (per-neuron, scaled by sd) at a given unexplained variance, to
show how much state error ARC tolerates. Saved to tcs_ft.pt."""
import sys, torch
sys.path.insert(0, "../../family")
from common import *
import score
from train_tc import TC, pairs, fvu, states
from eval_tc import load, spliced, sd
F32 = torch.float32
def run_tc(s, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=F32); Hs = []
    for t in range(10):
        h = s(torch.cat([h, X[:, t:t + 1].float()], 1))[0]; Hs.append(h)
    return torch.stack(Hs, 1)                         # scaled states
def finetune(s, lam, steps=3000, seed=0):
    g = torch.Generator(device=DEV).manual_seed(500 + seed); opt = torch.optim.Adam(s.parameters(), lr=3e-4)
    Wo = (ARC[2] * sd.to(dt)).float()                 # scores from scaled states
    for step in range(steps):
        X = torch.randn(4096, 10, device=DEV, dtype=dt, generator=g)
        with torch.no_grad(): Ht = states(X).float() / sd; St = scores(ARC, X).float()
        Hs = run_tc(s, X); Ss = Hs[:, -1] @ Wo.T
        mse = ((Hs - Ht) ** 2).sum(2).mean(); kl = torch.nn.functional.kl_div(Ss.log_softmax(1), St.log_softmax(1), log_target=True, reduction="batchmean")
        loss = mse + lam * kl; opt.zero_grad(); loss.backward(); opt.step()
    return s
def noisy(level, every):
    def f(X):
        g = torch.Generator(device=DEV).manual_seed(9); h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10):
            h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
            if every or t == 9: h = h + torch.randn(h.shape, device=DEV, dtype=dt, generator=g) * sd.to(dt) * level ** 0.5
        return h @ ARC[2].T
    return f
def rand_acc(f):
    g = torch.Generator(device=DEV).manual_seed(11); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    with torch.no_grad(): return 100 * (f(X).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item()
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(999); Xh = torch.randn(50000, 10, device=DEV, dtype=dt, generator=g)
    U, Y = pairs(Xh, sd); U, Y = U.reshape(-1, 17), Y.reshape(-1, 16)
    models = {}; out = {}
    for lvl in (0.007, 0.001):
        models[f"ARC + noise {100*lvl:.1f}% unexpl., every"] = [noisy(lvl, True)]; models[f"ARC + noise {100*lvl:.1f}% unexpl., last"] = [noisy(lvl, False)]
    for (m, k, lam) in [(32, 8, 1.0), (128, 8, 1.0), (128, 8, 10.0), (128, 4, 1.0)]:
        s0 = load(m, k)
        with torch.no_grad(): f0 = fvu(s0(U)[0], Y)
        s = finetune(load(m, k), lam)
        with torch.no_grad(): f1 = fvu(s(U)[0], Y); dead = ((s(U)[1] > 0).sum(0) == 0).sum().item()
        nm = f"TC {m}x k{k} ft lam {lam:g}"
        print(f"{nm}: single-step unexplained variance {100*f0:.2f}% before, {100*f1:.2f}% after; dead {dead}; random-list accuracy "
              f"every step {rand_acc(spliced(s0, True)):.1f}% -> {rand_acc(spliced(s, True)):.1f}%, last step {rand_acc(spliced(s0, False)):.1f}% -> {rand_acc(spliced(s, False)):.1f}%", flush=True)
        models[nm + ", every step"] = [spliced(s, True)]; models[nm + ", last step"] = [spliced(s, False)]
        out[(m, k, lam)] = {kk: v.detach().cpu() for kk, v in s.state_dict().items()}
    torch.save(out, "tcs_ft.pt")
    for nm, Ws in models.items():
        if nm.startswith("ARC"): print(f"{nm}: random-list accuracy {rand_acc(Ws[0]):.1f}%", flush=True)
    score.report(models, family=False)
