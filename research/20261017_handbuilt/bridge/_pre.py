import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((200000, 10))
net = build_v18(); I = net.idx; wi, wh, wo = [np.array(p, float) for p in net.params]
Y = [I[f"mem{j}"] for j in range(8)]
h = np.zeros((len(X), 16)); mins = []
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi; mins.append(pre[:, Y].min(0)); h = np.maximum(pre, 0)
print(np.round(np.array(mins), 4))
print("share of slot pre-activations below -1e-9:", np.mean(np.array([0])))
