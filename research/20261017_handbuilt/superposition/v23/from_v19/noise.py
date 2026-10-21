"""P7: robustness to relative weight noise, measured as in superposition/noise_control/run.py."""
import torch
from density import measure, T, DEV, dt
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    for name, W0 in (("v23 from v19 (1e-4)", T(build_v23f().params)), ("v23 from v19 (0.3)", T(build_v23f(ep=0.3).params)), ("v19", T(build_v19(kappa=0.5).params))):
        for sd in (0.001, 0.003, 0.01):
            for rep in range(2):
                gn = torch.Generator(device=DEV).manual_seed(1000 + rep)
                W = [w * (1 + sd * torch.randn(w.shape, device=DEV, dtype=dt, generator=gn)) for w in W0]
                measure(W, f"{name}, noise {sd:g} #{rep}")
