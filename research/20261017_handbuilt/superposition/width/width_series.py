"""Width as the variable. Trained second-argmax networks of 16 (ARC, the bo5 network, run-3 seed 209), 20 and 32
neurons (two). Per network, on random lists: (1) on/off patterns over all neurons (count, share covered by the top 30);
(2) switching simplicity: per neuron, depth-2 tree accuracy for on/off from list features (../patterns/switch_rules.py)
against always guessing the commoner state; reported as the share of the guessing error removed, mean over neurons, and
the share of neurons whose on/off a depth-2 tree gets to 97%; (3) readability: depth-2 regression tree R^2 of each
neuron's activity from the same features, mean (held out); (4) precision tolerance: accuracy with Gaussian noise added
to the final state at 0.5% and 2% of each neuron's variance (spliced at the last step only, and at every step);
(5) sparse autoencoders (TopK, width 2x and 4x the neurons, k 4 and 8): held-out unexplained variance and accuracy when
spliced at the last step and at every step."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, numpy as np, torch
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
sys.path.insert(0, "../../family"); sys.path.insert(0, "../patterns"); sys.path.insert(0, "../sae")
from common import *
from algzoo.core import load
from switch_rules import features
from train_sae import SAE
M = "/data/output/research/algzoo/models/extra/"
def nets():
    out = {"ARC (16)": ARC, "16 bo5": T(load(M + "2nd_argmax_16_10_1073741824_bo5.pth")), "16 run3-209": trained(209),
           "20": T(load(M + "2nd_argmax_20_1073741824.pth")), "32 bo5": T(load(M + "2nd_argmax_32_10_1073741824_bo5.pth")), "32 long": T(load(M + "2nd_argmax_32_17179869184.pth"))}
    return out
def run(W, X, hook=None):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); H = []
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if hook is not None and (hook[1] or t == 9): h = hook[0](h)
        H.append(h)
    return torch.stack(H, 1), h @ W[2].T
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(901); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g); y = X.argsort(1, descending=True)[:, 1]
    F = features(X)[:, 1:].clamp(-50, 50).reshape(-1, 8).cpu().numpy(); half = len(F) // 2
    for nm, W in nets().items():
        d = len(W[0]); H, S = run(W, X); acc = (S.argmax(1) == y).double().mean().item()
        P = (H[:, 1:] > 0); w = 2 ** torch.arange(d, device=DEV, dtype=torch.float64)
        codes = Counter(((P.double() * w).sum(-1)).flatten().cpu().numpy().tolist()); tot = sum(codes.values()); top30 = sum(v for _, v in codes.most_common(30)) / tot
        red, good, r2 = [], 0, []
        Hn = H[:, 1:].reshape(-1, d).cpu().numpy(); Pn = P.reshape(-1, d).cpu().numpy()
        for j in range(d):
            yy = Pn[:, j]; base = max(yy[half:].mean(), 1 - yy[half:].mean())
            if base > 0.999: red.append(1.0); good += 1
            else:
                a = (DecisionTreeClassifier(max_depth=2, min_samples_leaf=200).fit(F[:half], yy[:half]).predict(F[half:]) == yy[half:]).mean()
                red.append((a - base) / (1 - base)); good += a >= 0.97
            tr = DecisionTreeRegressor(max_depth=2, min_samples_leaf=200).fit(F[:half], Hn[:half, j]); p = tr.predict(F[half:])
            r2.append(1 - ((p - Hn[half:, j]) ** 2).mean() / max(Hn[half:, j].var(), 1e-12))
        sd = H.reshape(-1, d).std(0) + 1e-9
        noise = {}
        for lvl in (0.005, 0.02):
            for every in (False, True):
                gg = torch.Generator(device=DEV).manual_seed(5)
                hk = (lambda h, lvl=lvl: h + torch.randn(h.shape, device=DEV, dtype=dt, generator=gg) * sd * np.sqrt(lvl), every)
                noise[(lvl, every)] = (run(W, X, hk)[1].argmax(1) == y).double().mean().item()
        print(f"== {nm}: {d} neurons, accuracy {100*acc:.2f}% | patterns {len(codes)}, top 30 cover {100*top30:.1f}% | switching: depth-2 trees remove {100*np.mean(red):.0f}% of the guessing error, "
              f"{good}/{d} neurons at 97% | readability R2 {np.mean(r2):.2f} | noise on state 0.5%: last {100*noise[(0.005, False)]:.1f}%, every {100*noise[(0.005, True)]:.1f}%; 2%: last {100*noise[(0.02, False)]:.1f}%, every {100*noise[(0.02, True)]:.1f}%", flush=True)
        Hf = H.reshape(-1, d).float(); sdf = Hf.std(0) + 1e-6; Hs = Hf / sdf; tr, te = Hs[:1500000], Hs[1500000:]
        for m in (2 * d, 4 * d):
            for k in (4, 8):
                torch.manual_seed(m + k); s = SAE(m, k, d); s.bd.data = tr[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
                for step in range(4000):
                    b = tr[torch.randint(0, len(tr), (8192,), device=DEV)]; r, z = s(b); loss = ((r - b) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    r, _ = s(te[:300000]); fvu = (((r - te[:300000]) ** 2).sum() / ((te[:300000] - te[:300000].mean(0)) ** 2).sum()).item()
                    hk = lambda h: (s(h.float() / sdf)[0] * sdf).to(dt)
                    al = (run(W, X[:100000], (hk, False))[1].argmax(1) == y[:100000]).double().mean().item(); ae = (run(W, X[:100000], (hk, True))[1].argmax(1) == y[:100000]).double().mean().item()
                print(f"   SAE {m} wide, k {k}: unexplained {100*fvu:.2f}%, spliced last step {100*al:.1f}%, every step {100*ae:.1f}%", flush=True)
