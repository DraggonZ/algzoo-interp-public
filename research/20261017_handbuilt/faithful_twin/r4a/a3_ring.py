"""r4a analysis 3: (1) the late gate takes over the detector's backward writes (same function expected); how much the
network then needs it (late gate silenced at every step). (2) A memory ring (last delay cell back into the linear
route, weight w) and the memory stored spread across its neurons (dense block, level lam). Own accuracy, 1M lists;
memory eigenvalues of the commonest on/off pattern (steps 4-8) with memory participation, as score_mechanism.py."""
from collections import Counter
import numpy as np, torch, scipy.linalg as sl
from algzoo.handbuilt_v4a import build_v4a

DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]


def run(net, kill=None):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT); H = []
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
        if kill is not None: h[:, kill] = 0
        H.append(h)
    return 100 * ((h @ wo.T).argmax(1) == Y2).double().mean().item(), torch.stack(H, 1)


def modes(net, H, mem):
    W = np.asarray(net.params[1], float); on = (H[:200_000, 4:9] > 0).reshape(-1, W.shape[0]).cpu().numpy()
    pats = Counter(map(lambda q: q.tobytes(), on)); top_b, cnt = pats.most_common(1)[0]
    top = np.frombuffer(top_b, dtype=bool)
    ev, L, R = sl.eig(np.diag(top.astype(float)) @ W, left=True, right=True); out = []
    for k in range(len(ev)):
        if np.imag(ev[k]) <= 0: continue
        lr = np.conj(L[:, k]) * R[:, k]; pm = float(np.real(lr[mem].sum() / lr.sum()))
        out.append(f"{np.degrees(abs(np.angle(ev[k]))):.0f}deg|{abs(ev[k]):.2f}|part {pm:.2f}")
    off = [k for k, v in net.idx.items() if not top[v]]
    return f"commonest pattern {100 * cnt / len(on):.0f}% of steps, off: {off}; turning modes: " + ", ".join(out)


for name, kw in (("r4a-1 (detector writes backward wave)", {}), ("late gate writes backward wave", dict(bw_late=True)),
                 ("  + ring w 0.07", dict(bw_late=True, ring_w=0.07)),
                 ("  + ring w 0.07, spread lam 1e6", dict(bw_late=True, ring_w=0.07, lam=1e6))):
    net = build_v4a(**kw); acc, H = run(net)
    acc_k, _ = run(net, kill=net.idx["late"])
    mem = [net.idx[k] for k in net.idx if k.startswith("mem") or k == "linear"]
    print(f"{name}: accuracy {acc:.2f}, late gate silenced {acc_k:.2f}\n    {modes(net, H, mem)}")
