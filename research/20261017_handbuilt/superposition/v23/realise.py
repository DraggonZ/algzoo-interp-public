"""Density when v23 is realised as ARC realises its mechanism: memory spread across neurons on a lower level, and the
hiding neurons' gain lowered toward ARC's. Measures as in density.py."""
from density import measure, T
from algzoo.handbuilt_v23 import build_v23, spread_memory
if __name__ == "__main__":
    base = build_v23()
    for lam in (0.0, -1.0, -2.0, -2.5):
        measure(f"spread memory, shared level {4 + lam:.1f} m1", T(spread_memory(base, lam).params))
    for g in (100.0, 30.0, 10.0):
        measure(f"hiding gain {g:g}", T(build_v23(g=g).params))
    measure("hiding gain 10 + spread memory, level 2", T(spread_memory(build_v23(g=10.0), -2.0).params))
