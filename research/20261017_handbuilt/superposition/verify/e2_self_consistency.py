"""On ARC's wrong lists keep the top-3 values and positions; redraw the other seven
iid N(0,1) truncated below the third-largest; does ARC repeat the same wrong position?"""
from common import *
P = arc(); g = gen(9202)
x = draw(1_000_000, g); y = target(x); pr = predict(P, x)
w = pr != y; xw, prw = x[w], pr[w]; n = len(xw)
print(f"wrong lists: {n} of {len(x)}")
srt, idx = xw.sort(1, descending=True)
top3 = idx[:, :3]; a = srt[:, 2:3]                   # third-largest value
keep = torch.zeros_like(xw, dtype=torch.bool); keep.scatter_(1, top3, True)
print("ARC wrong pick is among top-3 positions:", f"{100*keep.gather(1, prw[:,None]).double().mean().item():.1f}%")
R = 20; same = []
for r in range(R):
    u = torch.rand(xw.shape, dtype=DT, device=DEV, generator=g)
    z = torch.special.ndtri(u * torch.special.ndtr(a))
    xr = torch.where(keep, xw, z)
    same.append((predict(P, xr) == prw).double())
S = torch.stack(same)                                # [R,n]
m = S.mean().item()
print(f"random redraw ({R} redraws per list): same wrong choice {100*m:.2f}% "
      f"(SE across lists {100*S.mean(0).std().item()/n**0.5:.2f})")
print(f"  share of lists where majority of redraws repeat it: {100*(S.mean(0)>0.5).double().mean().item():.2f}%")
# conditional-mean variant: the seven set to E[Z | Z < a]
cm = -torch.exp(-a**2/2)/(2*torch.pi)**0.5 / torch.special.ndtr(a)
xc = torch.where(keep, xw, cm.expand_as(xw))
c = (predict(P, xc) == prw).double().mean().item()
print(f"seven at conditional mean: same wrong choice {100*c:.2f}%")
k = keep.gather(1, prw[:, None])[:, 0]
print(f"breakdown, random redraw: wrong pick in top-3 -> {100*S[:,k].mean().item():.2f}%, "
      f"wrong pick among the seven -> {100*S[:,~k].mean().item():.2f}%")
cc = (predict(P, xc) == prw)
print(f"breakdown, conditional mean: top-3 -> {100*cc[k].double().mean().item():.2f}%, seven -> {100*cc[~k].double().mean().item():.2f}%")
