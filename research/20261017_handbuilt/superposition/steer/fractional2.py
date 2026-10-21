"""Why did rotating the block alone fail (fractional.log: a = 1 names second-1 on 23%, a full extra step 94.5%)? One
extra step in variants: (i) all 16 neurons, no ReLU; (ii) ReLU, only the rotating 12 updated (others frozen);
(iii) ReLU, all updated but the store's input to the rotating block cut; (iv) the full step."""
from harness import *
from modes import ROT, X, S, h
W = ARC[1]; f = lambda ans: 100 * (ans[S >= 1] == S[S >= 1] - 1).double().mean().item()
v1 = h @ W.T
v2 = h.clone(); v2[:, ROT] = torch.relu(h @ W[ROT].T)
W3 = W.clone(); W3[np.ix_(ROT, [2, 4])] = 0 if False else W3[np.ix_(ROT, [2, 4])]
import numpy as np
W3 = W.clone(); W3[torch.tensor(ROT, device=DEV)[:, None], torch.tensor([2, 4], device=DEV)[None]] = 0; v3 = torch.relu(h @ W3.T)
v4 = torch.relu(h @ W.T)
for lab, v in (("(i) all, no ReLU", v1), ("(ii) ReLU, rotating only", v2), ("(iii) ReLU, store cut from rotating", v3), ("(iv) full step", v4)):
    print(f"{lab:38s}: names second-1 {f((v @ ARC[2].T).argmax(1)):5.1f}%")
on = (v4 > 0).double().mean(0); print("share of lists on after the extra step, by neuron:", " ".join(f"{i}:{on[i]:.2f}" for i in range(16)))
