"""Substitution of the four hiding neurons by rules of growing detail (rulelib.py). Coefficients per (t, r, neuron)
fitted on 200,000 lists (seed 3); scored on 200,000 other lists (seed 4), all other neurons on their own weights."""
import numpy as np
from rulelib import fit_rule, substitute, report, lists, run
Xf, Xs = lists(3, 200_000), lists(4, 200_000)
report("network", Xs, run(Xs)[0].argmax(1))
for groups in (["PM"], ["PM", "age"], ["PM", "age", "left"], ["PM", "age", "left", "since"],
               ["PM", "age", "left", "since", "clip_own"], ["PM", "age", "left", "since", "clip"],
               ["PM", "age", "left", "since", "clip", "before"]):
    co = fit_rule(Xf, groups); report(" + ".join(groups), Xs, substitute(Xs, co, groups)[0].argmax(1))
