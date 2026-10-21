"""ARC neuron 3, zero (not average): remove only its direct output contribution at the end,
or only its recurrent outputs (into other neurons), per slot of the second."""
import torch as th

from algzoo.core import example

w_hi, w_hh, w_oh = (th.tensor(p, device="cuda") for p in example())
g = th.Generator(device="cuda").manual_seed(13)
x = th.randn(2_000_000, 10, generator=g, device="cuda", dtype=th.float64)
o = x.argsort(-1); tgt, p1 = o[:, -2], o[:, -1]


def run(direct_off, recur_off):
    Whh = w_hh.clone()
    if recur_off:
        Whh[:, 3] = 0; Whh[3, 3] = w_hh[3, 3]  # others stop reading 3 (3 keeps its own self-weight)
    Woh = w_oh.clone()
    if direct_off:
        Woh[:, 3] = 0
    h = th.zeros(len(x), 16, device="cuda", dtype=x.dtype)
    for t in range(10):
        h = th.relu(h @ Whh.T + x[:, t, None] * w_hi)
    return (h @ Woh.T).argmax(-1)


res = {k: run(*v) for k, v in {"normal": (0, 0), "direct off": (1, 0), "recurrent off": (0, 1), "both off": (1, 1)}.items()}
print("second's slot  " + "  ".join(f"{k:>13s}" for k in res))
for s in range(10):
    m = tgt == s
    print(f"     {s}         " + "  ".join(f"{(p[m]==s).double().mean().item():13.3f}" for p in res.values()))
print("overall        " + "  ".join(f"{(p==tgt).double().mean().item():13.3f}" for p in res.values()))
for k in ("direct off", "recurrent off"):
    p = res[k]; w = p != tgt
    print(f"{k}: wrong answers picking the largest {(p[w]==p1[w]).double().mean().item():.2f}")
