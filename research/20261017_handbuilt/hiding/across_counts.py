"""Counts from across.log, and how they depend on the cutoff chosen.
(A) networks (with hiding neurons) where undoing partial clipping one step after a record lowers accuracy on lists whose
    answer sits right after the largest by more than a cutoff (in points);
(B) networks with at least one hiding neuron whose reference follows the running second-largest: residual ratio
    (second-largest, largest) / (previous largest, largest) below a cutoff at K = 2 or 3."""
import re
txt = open("across.log").read().split("== ")[1:]
A, Bc, nets = {}, {}, []
for block in txt:
    name = block.split(":")[0]; lines = block.split("\n")
    if "hiding neurons []" in lines[0]: continue
    nets.append(name)
    for l in lines:
        m = re.search(r"undo at K1 : accuracy ([\d.]+) \(answer right after the largest ([\d.]+), normally ([\d.]+)\)", l)
        if m: A[name] = (float(m.group(3)) - float(m.group(2))) * 100
        if "(B) neuron" in l and "skipped" not in l:
            rs = [float(v) for v in re.findall(r"rms ratio ([\d.]+)", l)]
            Bc[name] = min(Bc.get(name, 9), min(rs))
print(f"{len(nets)} networks with hiding neurons")
for c in (1, 3, 5, 10, 30):
    ks = [k for k in nets if A.get(k, 0) > c]
    print(f"(A) cutoff {c:2d} points: {len(ks)} networks ({', '.join(ks)})")
for c in (0.3, 0.4, 0.5, 0.6, 0.7):
    ks = [k for k in nets if Bc.get(k, 9) < c]
    print(f"(B) cutoff {c:.1f}: {len(ks)} networks ({', '.join(ks)})")
