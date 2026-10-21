"""P4: forward-only hiding (no backward writes): is the ring then exact, and what does the uncompensated wave cost?"""
from check_basic import T, scores, lists, acc
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    X = lists(1_000_000, 2)
    base = scores(T(build_v23f(ep=0.1, bwd=False)), X).argmax(1)
    for ep in (0.1, 0.43, 1.0):
        a = scores(T(build_v23f(ep=ep, bwd=False)), X).argmax(1)
        print(f"wrap {ep}: accuracy {100*acc(T(build_v23f(ep=ep, bwd=False)), X):.2f}%, answers equal to wrap 0.1 on {100*(a == base).double().mean().item():.3f}%", flush=True)
