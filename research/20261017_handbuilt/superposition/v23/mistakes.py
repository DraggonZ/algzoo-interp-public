"""ARC's mistakes reproduced (shared scorer family/score.py: no-sampling grid; near ties by ratio and order) for v23
variants, v19 and the five-network family reference."""
import sys, os; sys.path.insert(0, os.path.abspath("../../family"))
os.chdir(os.path.abspath("../../family"))
from score import report, T
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v19 import build_v19
from leak_net import leaky
from nine_in_eight import null_dir
import numpy as np
if __name__ == "__main__":
    report({"v19 (kappa 0.5)": [T(build_v19(kappa=0.5).params)], "v23": [T(build_v23().params)],
            "v23, full loop": [T(build_v23(w=1.0).params)], "v23, hiding gain 10": [T(build_v23(g=10.0).params)],
            "v23 + ARC-angle leak": [T(leaky(build_v23(), null_dir([0, 43, 83, 126, 180])).params)]})
