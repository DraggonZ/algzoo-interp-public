import numpy as np
from algzoo.handbuilt_v23 import build_v23, scores
from algzoo.handbuilt_v19 import build_v19, scores as sc19
rng = np.random.default_rng(1); X = rng.standard_normal((200000, 10)); o = np.argsort(-X, 1); L, S, T = o[:, 0], o[:, 1], o[:, 2]
print("v19 k0.5", (sc19(build_v19(kappa=0.5), X).argmax(1) == S).mean())
for kw in (dict(), dict(kappa=0.0), dict(share=0.2), dict(share=0.8), dict(beta_f=0.0)):
    sc = scores(build_v23(**kw), X); y = sc.argmax(1); ok = y == S
    print(kw, f"acc {ok.mean():.4f}", "| wrong: names largest", f"{(y[~ok] == L[~ok]).mean():.2f}", "third", f"{(y[~ok] == T[~ok]).mean():.2f}")
sc = scores(build_v23(), X); y = sc.argmax(1); ok = y == S
pm = np.concatenate([np.zeros((len(X), 1)), np.maximum.accumulate(X, 1)[:, :-1]], 1); rec = X > np.maximum(pm, 0)
srec = rec[np.arange(len(X)), S]
for name, m in (("second was a record (revealed)", srec), ("second not a record", ~srec), ("second before largest", S < L), ("second after largest", S > L)):
    print(f"{name:32s} share {m.mean():.2f} acc {ok[m].mean():.4f}")
