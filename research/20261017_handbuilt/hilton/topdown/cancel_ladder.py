"""Residual ladder for the cancelling neurons (1, 6, 7, 8), scored on the grid lists (16 x 16 x 16, no sampling) by
mistakes reproduced, agreement and accuracy. Rules of growing detail from the hiding agent (../../hiding/rulelib.py,
fitted on 200,000 random lists, seed 3): each neuron's input is c (reference - x); the reference is written from
PM (previous largest and largest), + age (where the previous largest arrived), + left (the four's values at the record
step), + since (numbers since the record, and whether kept), + clip (the four's clipped amounts since the record)."""
import sys, os
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hiding"); os.chdir("/data/output/research/algzoo/20261017_handbuilt/hiding")
import numpy as np
from rulelib import fit_rule, substitute, lists
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/topdown"); import node_common as nc
Xf = lists(3, 200_000)
for groups in (["PM"], ["PM", "age"], ["PM", "age", "left"], ["PM", "age", "left", "since"], ["PM", "age", "left", "since", "clip"]):
    co = fit_rule(Xf, groups); ncoef = sum(np.size(c) for c in co.values())
    agr = acc = mw = zw = 0.0
    for i in range(0, len(nc.X), 100_000):
        s = slice(i, i + 100_000); Xc = nc.X[s]; a0 = nc.intact(Xc).argmax(1); a1 = substitute(Xc, co, groups)[0].argmax(1); wr = a0 != nc.Y[s]
        agr += (nc.w[s] * (a1 == a0)).sum(); acc += (nc.w[s] * (a1 == nc.Y[s])).sum(); mw += (nc.w[s] * (wr & (a1 == a0))).sum(); zw += (nc.w[s] * wr).sum()
    print(f"{' + '.join(groups):34s} ({ncoef:4d} numbers): mistakes reproduced {100*mw/zw:5.1f}%   agreement {100*agr:6.2f}%   accuracy {100*acc:6.2f}%", flush=True)
