"""Confusion of clean-neuron zero patterns against true events, per step (base model).

True events use the running max/2nd of (0, x_0..x_{t-1}) ('with 0') and of (x_0..x_{t-1}) ('no 0').
"""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import torch as th

from core import example, forward

dev = "cuda"
N = 2**19
with th.no_grad():
    x = th.randn(N, 10, device=dev, generator=th.Generator(dev).manual_seed(4))
    _, H = forward(example(), x, return_hidden=True)
    for with0 in (True, False):
        m1 = th.zeros(N, device=dev) if with0 else th.full((N,), -th.inf, device=dev)
        m2 = th.full((N,), -th.inf, device=dev)
        if with0:
            pass
        print("true events", "with phantom 0" if with0 else "without 0")
        for t in range(10):
            xt = x[:, t]
            newmax = xt > m1
            new2 = (~newmax) & (xt > m2)
            h = H[:, t + 1]
            z7, z8, p4 = h[:, 7] == 0, h[:, 8] == 0, h[:, 4] > 0
            z6, z1 = h[:, 6] == 0, h[:, 1] == 0
            def pr(a, b):
                return a[b].float().mean().item() if b.any() else float("nan")
            print(f"  t={t}: P(newmax) {newmax.float().mean():.3f} P(new2) {new2.float().mean():.3f} |"
                  f" n7=0: newmax {pr(z7,newmax):.3f} new2 {pr(z7,new2):.3f} ord {pr(z7,~newmax&~new2):.3f} |"
                  f" n8=0: newmax {pr(z8,newmax):.3f} new2 {pr(z8,new2):.3f} ord {pr(z8,~newmax&~new2):.3f} |"
                  f" n6=0: {pr(z6,newmax):.2f} {pr(z6,new2):.2f} {pr(z6,~newmax&~new2):.2f} |"
                  f" n1=0: {pr(z1,newmax):.2f} {pr(z1,new2):.2f} {pr(z1,~newmax&~new2):.2f}")
            m2 = th.where(newmax, m1, th.where(new2, xt, m2))
            m1 = th.where(newmax, xt, m1)
