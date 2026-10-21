"""How strongly do the seven smaller numbers move the near-tie decision? At near ties (third within 5% of second,
second > 0): per smaller number s, |J[S,s] - J[T,s]| (exact), divided by the network's own vote per unit (median of
J[S,S] and J[T,T]); median over lists and numbers, and the RMS over lists of the summed effect sum_s (J[S,s]-J[T,s]) x_s
in units of a vote. For ARC, trained 209 and 111, and v19 with input traces at several eta. The eta whose value
matches ARC's is the measured trace size (no use of mistakes)."""
import numpy as np, torch
from common import *
from traces import build
if __name__ == "__main__":
    X = near_lists(40000, seed=4); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    rest = torch.ones(len(X), 10, dtype=torch.bool, device=DEV)
    for p in (L, S, Tq): rest[ar, p] = False
    nets = {"ARC": ARC, "trained 209": trained(209), "trained 111": trained(111)}
    nets.update({f"v19 traces eta {e}": T(build(e)) for e in (0.0, 0.003, 0.01, 0.03, 0.1)})
    for nm, W in nets.items():
        J = jac(W, X); vote = 0.5 * (J[ar, S, S] + J[ar, Tq, Tq]).abs().median()
        dJ = (J[ar, S, :] - J[ar, Tq, :])
        per = (dJ.abs() / vote)[rest]; summed = ((dJ * X) * rest).sum(1) / vote
        print(f"{nm:24s} vote per unit {vote:9.3g} | per smaller number, median {per.median():.4f}, 90th pct {per.quantile(.9):.3f} | summed effect RMS {summed.pow(2).mean().sqrt():.3f} votes")
