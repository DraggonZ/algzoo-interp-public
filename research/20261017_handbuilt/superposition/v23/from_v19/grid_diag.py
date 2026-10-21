"""Diagnostic only (not used to pick weights): how the leftover's cost depends on the hiding margin kappa and the
flip share beta, at wrap 0.1 against no wrap."""
from check_basic import T, lists, acc
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    X = lists(300_000, 3)
    for kappa in (0.25, 0.5, 1.0, 2.0):
        for beta in (0.25, 0.5, 1.0):
            a0 = acc(T(build_v23f(ep=0, r1=0, kappa=kappa, beta=beta)), X)
            a1 = acc(T(build_v23f(ep=0.1, r1=0, kappa=kappa, beta=beta)), X)
            print(f"kappa {kappa:4.2f} beta {beta:4.2f}: no wrap {100*a0:.2f}%, wrap 0.1 {100*a1:.2f}% (cost {100*(a0-a1):.2f})", flush=True)
