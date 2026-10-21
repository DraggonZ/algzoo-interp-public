"""Level 1, memory node: neurons 0, 3, 10-15 as the memory agent's linear machine on the real weights, switched by its
fitted sign rules (../../memory/sign_rules3_fresh.py features, fitted on 200,000 lists seed 12), closed loop."""
import sys, os
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/memory"); os.chdir("/data/output/research/algzoo/20261017_handbuilt/memory")
src = open("sign_rules3_fresh.py").read().split("X = lists(99")[0]
exec(src)
from sign_rules2 import build_rules, apply_rules, machine
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/topdown"); import node_common as nc
R = build_rules(lists(12, 200_000))
def f(Xc):
    V = task2(Xc); S0, P0, H0 = run(Xc); off = apply_rules(R, Xc, V, H0); return machine(Xc, H0, off, "closed")
print(f"memory replaced by its linear machine with rule switching: {100 * nc.grid_acc(f):.2f}%", flush=True)
