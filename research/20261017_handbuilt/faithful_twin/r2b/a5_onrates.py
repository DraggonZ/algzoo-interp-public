"""Analysis: how often each delay cell and the linear route are off (steps 1-9), and the smallest ratio of the linear
route to delay cell 7, in the r2b-2 network (v23's order). Question: can delay cell 7 be added into the linear route
(z = L + nu y7, a change of basis) without clipping?"""
import numpy as np
from algzoo.handbuilt_v26 import build_v26, mix_memory

net = mix_memory(build_v26(), into="mem6", add="mem3", mu=1.0)
wi, wh, wo = [np.asarray(p, float) for p in net.params]
X = np.random.default_rng(3).standard_normal((200_000, 10)); h = np.zeros((len(X), 16)); P = []
for t in range(10):
    p = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(p, 0); P.append(p)
P = np.stack(P, 1)[:, 1:]
for k in ["linear"] + [f"mem{j}" for j in range(8)]:
    i = net.idx[k]; print(f"{k:7s} off {np.mean(P[:, :, i] < 0):.3f}  min pre {P[:, :, i].min():+.3f}")
L, Y7 = P[:, :, net.idx["linear"]], P[:, :, net.idx["mem7"]]
print("y7 / L: min", (Y7 / L).min(), " max", (Y7 / L).max(), " pre of y7 < 0 at steps:", np.mean(Y7 < 0, 0).round(3))
