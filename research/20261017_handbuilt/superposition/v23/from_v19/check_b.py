"""P5: the second layout against v19."""
from check_basic import T, scores, lists, acc
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    X = lists(10_000, 1)
    v19 = scores(T(build_v19(kappa=0.5)), X)
    for ring in (False, True):
        d = (scores(T(build_v23f(r1=0.0, ring=ring)), X) - v19).abs()
        print(f"P5a ring={ring}: max |difference| positions 0-8 {d[:, :9].max().item():.2e}, position 9 {d[:, 9].max().item():.2e}")
    X = lists(1_000_000, 2)
    print(f"v19 {100*acc(T(build_v19(kappa=0.5)), X):.2f}%")
    for lab, kw in (("gate route only", dict(r1=0.0)), ("half direct (default)", {}), ("half direct, loop gain 0.3", dict(ep=0.3)),
                    ("half direct, loop gain 0.3, forward-only hiding", dict(ep=0.3, bwd=False))):
        print(f"{lab:48s} {100*acc(T(build_v23f(**kw)), X):.2f}%", flush=True)
