import sys, importlib.util
from collections import Counter
import numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_twin import lists, run, so_far, T, N, HID, MEM, DEV, DT, example
def load(p):
    if p == "ARC": return N(example())
    s = importlib.util.spec_from_file_location("c", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return N(m.weights())
xs = torch.linspace(-1.0, 4.0, 5001, device=DEV, dtype=DT)
def offpoint(W, P, M, k):
    """record M at step 2 over previous largest P at step 1 (x0 = -1); test x at step 2+k; smallest x at which each hiding neuron is off at that step."""
    Wt = T(W); n = len(xs); X = torch.full((n, 10), -1.0, device=DEV, dtype=DT); X[:, 1] = P; X[:, 2] = M; X[:, 2 + k] = xs
    H, _ = run(Wt, X); out = []
    for j in HID:
        off = H[:, 2 + k, j] <= 0; i = torch.nonzero(off).flatten(); out.append(xs[i[0]].item() if len(i) else np.nan)
    return out
for p in sys.argv[1:]:
    W = load(p); print("==", p)
    for k in (1, 2, 3):
        rows = []
        for P in (0.5, 0.9, 1.3):
            for M in (1.6, 2.0, 2.4): rows.append((P, M, offpoint(W, P, M, k)))
        A = np.array([[P, M, 1] for P, M, _ in rows]); Y = np.array([o for *_, o in rows])
        cf = [np.linalg.lstsq(A[~np.isnan(Y[:, i])], Y[~np.isnan(Y[:, i]), i], rcond=None)[0] if (~np.isnan(Y[:, i])).sum() > 3 else [np.nan] * 3 for i in range(4)]
        print(f"  {k} step(s) after the record: off point = a x previous largest + b x largest: " + "  ".join(f"{j}: {c[0]:.2f}P+{c[1]:.2f}M" for j, c in zip(HID, cf)))
    # effective dynamics in the commonest on-pattern
    Wt = T(W); X = lists(200000, 7); H, _ = run(Wt, X); on = (H > 0)
    pats = Counter(map(lambda q: q.tobytes(), on[:, 4:9].reshape(-1, 16).cpu().numpy())); top = np.frombuffer(pats.most_common(1)[0][0], dtype=bool)
    sig = H.reshape(-1, 16).std(0).cpu().numpy(); sig = np.where(sig > 1e-12, sig, 1.0)
    A = np.diag(top.astype(float)) @ (W[1] * sig[None, :] / sig[:, None]); ev, V = np.linalg.eig(A)
    print("  on-pattern", "".join("1" if b else "0" for b in top), f"share {pats.most_common(1)[0][1] / (5 * len(X)):.2f}")
    for l, v in sorted(zip(ev, V.T), key=lambda z: -abs(z[0])):
        ang = np.degrees(abs(np.angle(l)))
        if abs(l) >= 0.3 and np.imag(l) >= 0: print(f"   eig |{abs(l):.2f}| {ang:5.1f} deg  memory share {np.sum(np.abs(v[MEM]) ** 2) / np.sum(np.abs(v) ** 2):.2f}  hiding share {np.sum(np.abs(v[HID]) ** 2) / np.sum(np.abs(v) ** 2):.2f}")
    # neuron 3 used: cut 3 -> everything, accuracy on lists with gap > 1, and share naming the largest
    o = X.argsort(1, descending=True); v = X.gather(1, o); big = (v[:, 0] - v[:, 1]) > 1.0
    for name, Wx in (("intact", W), ("3's outputs cut", [W[0], W[1] * (np.arange(16) != 3)[None, :] + np.diag(np.zeros(16)), W[2]])):
        if name != "intact": Wx[1][3, 3] = W[1][3, 3]
        Wt2 = T(Wx); H2, _ = run(Wt2, X); pred = (H2[:, -1] @ Wt2[2].T).argmax(1)
        print(f"  {name}: gap>1 lists right {100 * (pred == o[:, 1])[big].double().mean().item():.1f}%, naming the largest {100 * (pred == o[:, 0])[big].double().mean().item():.1f}%; all lists {100 * (pred == o[:, 1]).double().mean().item():.1f}%")
