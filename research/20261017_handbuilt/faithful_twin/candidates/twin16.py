"""Baseline candidate: the v23 twin with ARC's settings (U_mem=1, share=0.25, rho=0.99, level_exact=False), matched into
ARC's neuron slots by exact relabelling and rescaling (../../superposition/twin/match_twin.py)."""
import numpy as np
def weights():
    d = np.load("/data/output/research/algzoo/20261017_handbuilt/superposition/twin/twin16_matched.npz")
    return d["wi"], d["wh"], d["wo"]
