"""P12: ARC's mistakes reproduced and near-tie behaviour, with family/score.py."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
from common import T
import score
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    models = {"v19": [T(build_v19(kappa=0.5).params)]}
    for ep in (1e-4, 0.03, 0.1, 0.3):
        models[f"v23 from v19, loop gain {ep:g}"] = [T(build_v23f(ep=ep).params)]
    score.report(models)
