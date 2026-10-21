"""(1) Piece cap: cost and bias of stopping each end's walk after P pieces (the stretch then counts up to the cap).
(2) Control: plain quasi-random sampling (same scrambled Sobol family, 10 normals per list, network's 0/1 answer) at
the same operation count as TLQ m = 1 and 4. Is TLQ's gain from the line integrals or from the point set?"""
import sys, numpy as np, torch, time
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
import tlq
from linelib import F_LIST, DEV, dt
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
nets = (("ARC", ARC, tr[0]), ("203", trained(203), tr[20]), ("208", trained(208), tr[25]), ("114", trained(114), tr[15]))
for name, W, t in nets:
    for cap in (1, 2, 3, 40):
        accs = []
        for seed in range(8): a, fl = tlq.estimate(W, 1, seed, max_pieces=cap); accs.append(a)
        accs = np.array(accs)
        print(f"{name} TLQ m 1 cap {cap}: bias {100*(accs.mean()-t):+.3f}, SD {100*accs.std():.3f}, RMS {100*np.sqrt(np.mean((accs-t)**2)):.3f}, cost {fl/F_LIST:.0f} lists, sampling SE {100*np.sqrt(t*(1-t)*F_LIST/fl):.3f}", flush=True)
    for m in (1, 4):
        _, fl = tlq.estimate(W, m, 0); N = int(fl / F_LIST); accs = []
        for seed in range(8):
            eng = torch.quasirandom.SobolEngine(10, scramble=True, seed=100 + seed)
            X = torch.special.ndtri(eng.draw(N).to(DEV, dt).clamp(1e-12, 1 - 1e-12))
            accs.append((scores(W, X).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item())
        accs = np.array(accs)
        print(f"{name} quasi-random lists, N {N} (= TLQ m {m} cost): bias {100*(accs.mean()-t):+.3f}, SD {100*accs.std():.3f}, RMS {100*np.sqrt(np.mean((accs-t)**2)):.3f}; plain sampling SE {100*np.sqrt(t*(1-t)/N):.3f}", flush=True)
