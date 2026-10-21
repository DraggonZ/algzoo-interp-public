exec(open("proto.py").read().split("pairs = ")[0])
pairs = [k for k in range(8) if 1e-6 < np.angle(ev[k]) < 3.1]
slow = [k for k in pairs if abs(np.degrees(np.angle(ev[k])) - 38.3) < 1][0]
def run(x):
    h = np.zeros(16); J = np.zeros((16, 10)); hs=[h]; Js=[J]; Ds=[]
    for t in range(10):
        pre = wh @ h + wi * x[t]; D = (pre > 0).astype(float)
        e = np.zeros(10); e[t] = 1
        J = D[:, None] * (wh @ J + np.outer(wi, e)); h = D * pre
        hs.append(h); Js.append(J); Ds.append(pre)
    return hs, Js, Ds
rng = np.random.default_rng(1)
# input to neuron j from memory = wh[j,MEM] @ h_mem = sum_k (wh[j,MEM] V[:,k]) z_k ; mark for neuron j in mode k: direction where this is max: -arg(wh[j,MEM]V[:,k])
for j in [0, 10, 11, 12, 14, 15, 3]:
    out = []
    for age in range(0, 9):
        devs = []
        for n in range(1500):
            x = rng.standard_normal(10); r = rng.integers(0, 9 - age)
            # make position r the largest so far and stays largest until r+age
            x[r] = abs(x).max() + 0.3
            hs, Js, Ds = run(x)
            t = r + 1 + age
            if t > 10: continue
            if x[:t].argmax() != r: continue
            c = Js[t][MEM, r] * x[r]
            z = U[slow] @ c
            d = np.angle(z * (wh[j, MEM] @ V[:, slow]))
            devs.append(np.degrees(d))
        devs = np.array(devs)
        out.append(f"{np.median(np.abs(devs)):4.0f}")
    print(f"neuron {j:2d}: |angle between largest's slow arrow and neuron's mark| by age 0..8:", " ".join(out))
