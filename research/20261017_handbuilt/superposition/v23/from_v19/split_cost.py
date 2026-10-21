"""Which change costs what: keep gate level, direct route, wrap; and where the wrap's mistakes are."""
import torch
from check_basic import T, scores, lists, acc, DEV, dt
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    X = lists(1_000_000, 2)
    for lab, kw in (("v19 rules, gate level 0.5", dict(ep=0, r1=0, gate_level=0.5)),
                    ("gate level 4", dict(ep=0, r1=0)),
                    ("wrap 0.1, gate only", dict(ep=0.1, r1=0)),
                    ("wrap 0.1, half direct", dict(ep=0.1, r1=0.5)),
                    ("wrap 0.1, all direct", dict(ep=0.1, r1=1.0))):
        print(f"{lab:28s} {100*acc(T(build_v23f(**kw)), X):.2f}%", flush=True)
    # where do the wrap's extra mistakes fall? by position of largest (L) and second (S), and number of records
    o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]
    cm = torch.cummax(torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), X], 1), 1).values
    rec = X > cm[:, :-1]; nrec = rec.sum(1)
    a0 = scores(T(build_v23f(ep=0, r1=0)), X).argmax(1) == S
    a1 = scores(T(build_v23f(ep=0.1, r1=0)), X).argmax(1) == S
    lost = a0 & ~a1; gained = a1 & ~a0
    print(f"lost {100*lost.double().mean().item():.2f}%, gained {100*gained.double().mean().item():.2f}%")
    print("lost by order: second before largest", f"{100*(lost & (S < L)).double().mean().item():.2f}%", "after", f"{100*(lost & (S > L)).double().mean().item():.2f}%")
    for n in range(1, 6):
        m = nrec == n
        print(f"  {n} records ({100*m.double().mean().item():4.1f}% of lists): lost {100*lost[m].double().mean().item():.2f}%")
    # what does the network name instead, on lost lists?
    ans = scores(T(build_v23f(ep=0.1, r1=0)), X).argmax(1)
    print(f"  on lost lists it names the largest {100*(ans[lost] == L[lost]).double().mean().item():.0f}%, the third {100*(ans[lost] == o[lost, 2]).double().mean().item():.0f}%, other {100*((ans[lost] != L[lost]) & (ans[lost] != o[lost, 2])).double().mean().item():.0f}%")
    print(f"  named position after the largest: {100*(ans[lost] > L[lost]).double().mean().item():.0f}%; the second after the largest: {100*(S[lost] > L[lost]).double().mean().item():.0f}%")
