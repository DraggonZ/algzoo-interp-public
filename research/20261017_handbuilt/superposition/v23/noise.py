"""Robustness: relative weight noise (each weight x (1 + sd z)) as in noise_control/run.py, for v23 at its default gains
and at lower gains (detector and store 10, hiding 100; ARC's scale is about 10)."""
import torch
from density import measure, T, DEV, dt
from algzoo.handbuilt_v23 import build_v23
if __name__ == "__main__":
    for name, net in (("v23 default gains", build_v23()), ("v23 gains a 10, s 10, g 100", build_v23(a=10.0, s=10.0, g=100.0))):
        W0 = T(net.params); measure(f"{name}, no noise", W0)
        for sd in (0.001, 0.003, 0.01):
            gn = torch.Generator(device=DEV).manual_seed(1000)
            measure(f"{name}, noise {sd:g}", [w * (1 + sd * torch.randn(w.shape, device=DEV, dtype=dt, generator=gn)) for w in W0])
