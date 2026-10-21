"""Fair votes (README experiment 12): K[q, p] = W_out[q] . V[:, p]; is K Toeplitz on positions 0-8, and what does
imposing it cost? Two versions of V: linear (most common on/off pattern) and empirical (mean d h_final / d x_p on lists
where p holds the second-largest). Readout change: least-norm dW with dW V = T - K on the 9x9 block (rows centred
readout; columns of other positions held fixed)."""
from collections import Counter
import numpy as np
import torch as th
from algzoo.core import example, forward

W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
Xs = np.random.default_rng(17).standard_normal((100_000, 10)); h = np.zeros((len(Xs), 16)); masks = []
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0); masks.append(h > 0)
top, _ = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
D = np.diag(np.frombuffer(top, dtype=bool).astype(float)); M = D @ wh
V_lin = np.stack([np.linalg.matrix_power(M, 9 - p) @ D @ wi for p in range(10)], 1)

x = th.randn(200_000, 10, dtype=th.float64, generator=th.Generator().manual_seed(9), requires_grad=True)
_, H = forward(W0, x, return_hidden=True)
hf = H[:, -1]
J = th.stack([th.autograd.grad(hf[:, i].sum(), x, retain_graph=True)[0] for i in range(16)], 1).detach().numpy()  # [n,16,10]
second = x.detach().argsort(-1)[:, -2].numpy()
V_emp = np.stack([J[second == p, :, p].mean(0) for p in range(10)], 1)
print("lists per position (second-largest there):", np.bincount(second))

dev = "cuda"; N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(wo_new):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in (wi, wh, wo_new))
    hh = th.zeros(N, 16, device=dev)
    for t in range(10):
        hh = th.relu(hh @ b.T + X[:, t, None] * a)
    return ((hh @ c.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(wo)
print(f"ARC {acc0:.5f}")
np.set_printoptions(precision=2, suppress=True, linewidth=200)
Wc = wo - wo.mean(0)
for name, V in [("linear", V_lin), ("empirical", V_emp)]:
    K = Wc @ V
    B = K[:9, :9]
    T = np.zeros_like(B)
    for d in range(-8, 9):
        idx = np.arange(max(0, -d), min(9, 9 - d))
        T[idx, idx + d] = B[idx, idx + d].mean()
    diag = np.diag(B)
    print(f"\n{name} K (rows: scored position, columns: position of the number):\n{K}")
    print(f"  diagonal over 0-8: {diag}  (mean {diag.mean():.2f}, spread +-{100 * (diag.max() - diag.min()) / 2 / diag.mean():.0f}%)")
    print(f"  distance of the 9x9 block from Toeplitz: {np.linalg.norm(B - T) / np.linalg.norm(B):.3f} of its size")
    # least-norm readout change: rows q=0..8 get dW[q] with dW[q] V[:, :9] = (T - B)[q]; also keep columns 9 fixed
    Vk = V[:, :10]
    tgt = np.concatenate([T - B, np.zeros((9, 1))], 1)  # change on columns 0-8, none on column 9
    dW = np.zeros_like(wo)
    dW[:9] = tgt @ np.linalg.pinv(Vk)
    print(f"  readout change: {np.linalg.norm(dW) / np.linalg.norm(Wc):.3f} of the readout's size; cost {100 * (acc0 - acc(wo + dW)):.2f} points")
    for f in [0.1, 0.25, 0.5]:
        print(f"    a fraction {f} of that change: cost {100 * (acc0 - acc(wo + f * dW)):.2f} points")
