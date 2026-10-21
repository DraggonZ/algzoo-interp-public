"""Two trivial decompositions to compare SPD against, on 2^20 random lists (exactly faithful, KL 0):
- neuron basis: the recurrent and readout matrices split into their 16 columns (piece j reads neuron j), the input weights
  one piece. Piece j is needed only when neuron j is on, so it is active at a step iff neuron j > 0 there.
- singular value decomposition: 16 + 10 + 1 rank-one pieces, all active always.
Counts are in the same units as run.py's eval (uses per list summed over steps, distinct pieces per list)."""
import sys
from pathlib import Path

import torch as th

sys.path.insert(0, str(Path(__file__).parent))
from spd_rnn import N_STEPS, arc_params, run_target  # noqa: E402

W = arc_params("cuda")
g = th.Generator(device="cuda").manual_seed(12345)
tot = {"uses_hh": 0.0, "distinct_hh": 0.0, "uses_oh": 0.0, "per_step": th.zeros(N_STEPS - 1, device="cuda")}
n = 2**20
for _ in range(16):
    x = th.randn(n // 16, N_STEPS, generator=g, device="cuda")
    _, hs = run_target(W, x)
    on = (hs[:, 1:N_STEPS] > 0).float()  # inputs to the recurrent matrix at steps 1..9
    tot["uses_hh"] += on.sum().item()
    tot["distinct_hh"] += on.amax(1).sum().item()
    tot["uses_oh"] += (hs[:, N_STEPS] > 0).float().sum().item()
    tot["per_step"] += on.sum((0, 2))
print("neuron basis (exact, KL 0):")
print(f"  recurrent piece-uses per list {tot['uses_hh'] / n:.2f} (of 16 x 9 = 144); distinct per list {tot['distinct_hh'] / n:.2f}")
print(f"  on neurons per step 1..9: {[round(v, 2) for v in (tot['per_step'] / n).tolist()]}")
print(f"  readout pieces per list {tot['uses_oh'] / n:.2f}; input pieces per list 10 uses, 1 distinct")
print("singular value decomposition (exact, KL 0): recurrent 16 x 9 = 144 uses, readout 10, input 10 uses")
