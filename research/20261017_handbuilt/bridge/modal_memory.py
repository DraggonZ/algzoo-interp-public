"""Rewrite v18's slot memory in rotating coordinates, exactly. New memory neurons
   z = L m1 + T (Y - lam m1),
T = the real Fourier basis over the 8 slots (constant, cos/sin at 45, 90 and 135 degrees per slot, alternating).
m1 = store/20 (the largest before this step, the level the slots carry) is linear in the state and never clips, so z = M h for a fixed invertible M, and the network
h' = M h has weights wi' = M wi, wh' = M wh M^-1, wo' = wo M^-1. This is the same function as long as no memory
neuron clips, before or after: v18's slots never go below zero (their level covers them), and L is set to the
smallest multiple of m1 that keeps every z at or above zero on these lists, plus a margin.
Then: is the memory block's look and its readout (dials) closer to the trained network's?"""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
def run(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi))); hs = []
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return h @ wo.T, np.stack(hs, 1)
net = build_v18(); I = net.idx; P = [np.array(p, float) for p in net.params]
Y = [I[f"mem{j}"] for j in range(8)]; S, D = I["store"], I["detect"]; lam = 10.0
lg, H = run(P, X)
print("v18 accuracy", (lg.argmax(1) == y).mean(), " smallest slot value", H[:, :, Y].min())
j = np.arange(8); T = [np.ones(8) / np.sqrt(8)]
for k in (1, 2, 3):
    T += [np.cos(2 * np.pi * k * j / 8) / 2, np.sin(2 * np.pi * k * j / 8) / 2]
T += [(-1.0) ** j / np.sqrt(8)]; T = np.array(T)                       # orthonormal rows
m1 = np.zeros(16); m1[S] = 1 / 20          # the store holds the largest before this step, which is the slots' level; it never clips
def build(L):
    M = np.eye(16)
    for r, row in zip(Y, T):
        M[r] = 0; M[r, Y] = row; M[r] -= lam * row.sum() * m1; M[r] += L * m1
    Mi = np.linalg.inv(M)
    return M, (M @ P[0], M @ P[1] @ Mi, P[2] @ Mi)
M0, _ = build(0.0)
Z0 = H @ M0.T                                                           # modal values with L = 0
m1v = H @ m1
need = np.where(m1v[..., None] > 1e-9, -Z0[..., Y] / np.maximum(m1v[..., None], 1e-9), 0).max(axis=(0, 1))
print("level needed per modal neuron (units of m1):", np.round(need, 2))
L = 2.0 * need.max()                                                # margin for lists not in this sample
M, Q = build(L)
lq, HQ = run(Q, X)
print(f"L = {L:.2f}: accuracy", (lq.argmax(1) == y).mean(), " max |score change|", np.abs(lq - lg).max(), " smallest modal value", HQ[:, :, Y].min())
mb = Q[1][np.ix_(Y, Y)]; ev = np.linalg.eigvals(mb)
print("modal memory block eigenvalues |.|", np.round(np.abs(ev), 3))
np.savez("modal_v18.npz", wi=Q[0], wh=Q[1], wo=Q[2], L=L)
