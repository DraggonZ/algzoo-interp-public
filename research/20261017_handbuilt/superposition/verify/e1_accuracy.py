from common import *
N = 2_000_000
x = draw(N, gen(9101)); y = target(x)
for name, p in [("ARC", arc()), ("v18", v18())]:
    a = (predict(p, x) == y).double().mean().item()
    print(f"{name}: accuracy {100*a:.3f}% (+-{se(a,N):.3f} SE) on {N} lists")
