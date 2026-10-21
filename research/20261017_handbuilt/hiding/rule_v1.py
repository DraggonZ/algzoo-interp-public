"""Rule V1: c_n ref_n(t) = [fitted: P, M, P's arrival, numbers since (plain and kept)] + [fixed: the four's own block
carrying the leftover at the record step and every clipped amount since]. Fit seed 3, score seed 4, 200,000 lists each.
Then the same with the fitted part pooled over records at different positions (coefficients depend on K only) and
with parts removed, to see which terms the accuracy needs."""
import numpy as np
from rulelib import fit_rule, substitute, report, lists, run, own_echo, own_echo_full
Xf, Xs = lists(3, 200_000), lists(4, 200_000)
report("network", Xs, run(Xs)[0].argmax(1))
for groups in (["PM", "age", "since"], ["PM", "since"], ["PM", "age"], ["PM"]):
    co = fit_rule(Xf, groups, fixed=own_echo_full)
    report("V1 fitted " + " + ".join(groups) + " | fixed own echo", Xs, substitute(Xs, co, groups, fixed_step=own_echo)[0].argmax(1))
