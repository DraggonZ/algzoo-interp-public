"""P11: close-call density against the ring's loop gain (same measures as density.py)."""
from density import measure, T
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    for ep in (0.01, 0.03, 0.1, 0.3, 1.0):
        measure(T(build_v23f(ep=ep).params), f"v23 from v19, loop gain {ep}")
