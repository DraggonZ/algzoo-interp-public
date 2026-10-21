"""Level 1, cancelling-neuron node: neurons 1, 6, 7, 8 replaced by the hiding agent's best rule (../../hiding/rulelib.py,
features PM + age + left + since + clip, fitted on 200,000 random lists, seed 3), the rest on their own weights."""
import sys, os
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hiding"); os.chdir("/data/output/research/algzoo/20261017_handbuilt/hiding")
from rulelib import fit_rule, substitute, lists
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/topdown"); import node_common as nc
G = ["PM", "age", "left", "since", "clip"]; co = fit_rule(lists(3, 200_000), G)
print(f"intact network on the grid: {100 * nc.grid_acc(nc.intact):.2f}%", flush=True)
print(f"cancelling neurons replaced by their rule: {100 * nc.grid_acc(lambda Xc: substitute(Xc, co, G)[0]):.2f}%")
