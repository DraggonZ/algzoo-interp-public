"""Level 1, gate nodes: keep gate (5) and late gate (9) by the gates agent's rules (../../gates/g28_final_rules.py:
keep gate pooled, late gate short rule and tables; coefficients from g28_tables.pkl), each alone and both."""
import sys, os, pickle
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gates"); os.chdir("/data/output/research/algzoo/20261017_handbuilt/gates")
from gatelib import run
from gaterules3 import rule_values, HJ
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/topdown"); import node_common as nc
co = pickle.load(open("g28_tables.pkl", "rb"))
D5K = "gates|-kept|-forgot|-neg|-rec|-p"
keep_pooled = {**{f"r{j}": "ages" for j in HJ}, "D5": D5K, "D9": "ages"}
late_short = {"r1": "off", "r4": "basic", "r6": "off", "r7": "const:1.0", "r8": "off", "D9": "basic", "D5": "basic"}
late_table = {**{f"r{j}": "ages" for j in HJ}, "D9": "ages", "D5": "basic"}
def sub(specs):
    def f(Xc):
        s = {g: rule_values(Xc, sp, co)[0][g] for g, sp in specs}
        return run(Xc, sub=s)[0]
    return f
for name, specs in [("keep gate (pooled rule)", [(5, keep_pooled)]), ("late gate (short rule)", [(9, late_short)]),
                    ("late gate (tables)", [(9, late_table)]), ("both gates (keep pooled + late short)", [(5, keep_pooled), (9, late_short)])]:
    print(f"{name}: {100 * nc.grid_acc(sub(specs)):.2f}%", flush=True)
