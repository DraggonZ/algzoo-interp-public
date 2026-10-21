"""Where does the wrap's leftover (an overtaken record's wave in slots written after the overtaking) cost accuracy?
Leftover = scores(wrap 0.1) - scores(no wrap), both with the gate route only, so nothing else differs."""
import torch
from check_basic import T, scores, lists, DEV, dt
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    X = lists(1_000_000, 2)
    o = X.argsort(1, descending=True); L, S, Tt = o[:, 0], o[:, 1], o[:, 2]
    n = torch.arange(len(X), device=DEV)
    s0 = scores(T(build_v23f(ep=0, r1=0)), X); s1 = scores(T(build_v23f(ep=0.1, r1=0)), X)
    J = s1 - s0
    pos = torch.arange(10, device=DEV)[None, :]
    print(f"no wrap {100*(s0.argmax(1) == S).double().mean():.2f}%, wrap 0.1 {100*(s1.argmax(1) == S).double().mean():.2f}%")
    print(f"leftover nonzero at positions after the largest only? max |leftover| before the largest: {J[pos < L[:, None]].abs().max().item():.2e}")
    for lab, m in (("at the largest", pos == L[:, None]), ("after the largest, not the largest", pos > L[:, None]),
                   ("at the second", pos == S[:, None]), ("at the third", pos == Tt[:, None])):
        a = ((s1 - J * m).argmax(1) == S).double().mean().item()
        print(f"remove the leftover {lab:36s}: {100*a:.2f}%")
    # size of the leftover at the largest, per unit of the largest's hiding margin
    jl = J[n, L]
    print(f"leftover at the largest: positive on {100*(jl > 1e-9).double().mean():.0f}% of lists, median {jl[jl.abs() > 1e-9].median().item():.3f}")
