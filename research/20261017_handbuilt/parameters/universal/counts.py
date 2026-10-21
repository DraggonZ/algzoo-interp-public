"""Cutoffs per relation, counts over the 37 and over the nine most accurate of them (accuracy >= 94.5%; this split was
chosen after seeing that tidiness tracks accuracy), and each quantity's rank correlation with accuracy over the 37.
The first cutoff of each relation is the one written in the README's predictions; the stricter second ones were added
afterwards to describe the spread, not as tests. Writes counts.log."""
import csv, numpy as np
from scipy.stats import spearmanr
rows = list(csv.DictReader(open("quantities.csv"))); arc = rows[0]; oth = rows[1:]
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
tests = {
 "readout shift: median cosine >= 0.8": lambda r: f(r, "readout shift: median cosine (k=0..7)") >= 0.8,
 "readout shift: median cosine >= 0.95": lambda r: f(r, "readout shift: median cosine (k=0..7)") >= 0.95,
 "memory has rotating pairs, largest modulus 0.8-1.2": lambda r: f(r, "memory: number of rotating pairs") >= 1 and 0.8 <= f(r, "memory: largest modulus") <= 1.2,
 "at least one age-clipped memory neuron": lambda r: f(r, "memory: age-clipped neurons") >= 1,
 "blindness: median ratio < 0.1 (null about 0.5)": lambda r: f(r, "blindness: median ratio (0 = blind)") < 0.1,
 "blindness: median ratio < 0.02": lambda r: f(r, "blindness: median ratio (0 = blind)") < 0.02,
 "hiding references: fit R2 >= 0.8": lambda r: f(r, "hiding: reference fit R2 (median)") >= 0.8,
 "(a) detector row = -c x store row: |cosine| >= 0.9": lambda r: f(r, "(a) detector row vs store row: cosine") <= -0.9,
 "(a) ... and c within 20% of the overwrite value": lambda r: f(r, "(a) detector row vs store row: cosine") <= -0.9 and 0.8 <= f(r, "(a) c fitted / c from overwrite") <= 1.2,
 "(b) a hiding row mirrors the detector: cosine <= -0.95": lambda r: f(r, "(b) mirror: most negative cosine with detector") <= -0.95,
 "(c) hiding store ratio = detector's (median within 0.7-1.3)": lambda r: 0.7 <= f(r, "(c) hiding store ratio / detector's (median)") <= 1.3,
 "(c) ... for every hiding neuron": lambda r: f(r, "(c) share of hiding neurons within 0.7-1.3") == 1,
 "(d) delay line: a memory row with one source >= 0.8": lambda r: f(r, "(d) delay line: largest one-source share in a memory row") >= 0.8,
 "(d) delay line: one source >= 0.95": lambda r: f(r, "(d) delay line: largest one-source share in a memory row") >= 0.95,
 "(e) readout column |mean|/sd below 0.2 (random rows about 0.3)": lambda r: f(r, "(e) readout column |mean| / sd (rms)") < 0.2,
 "(e) below 0.05": lambda r: f(r, "(e) readout column |mean| / sd (rms)") < 0.05,
 "clear store (correlation >= 0.9)": lambda r: f(r, "store: correlation with largest") >= 0.9,
 "store self-weight >= 0.9": lambda r: f(r, "store: self-weight (fade per step)") >= 0.9,
 "strict detector": lambda r: f(r, "strict detector") >= 1,
 "hiding group of 3-6": lambda r: 3 <= f(r, "hiding size") <= 6,
 "late-like gate": lambda r: f(r, "late gate") >= 1, "keep-like gate": lambda r: f(r, "keep gate") >= 1,
 "third is the commonest wrong answer": lambda r: f(r, "mistakes: third") > max(f(r, "mistakes: largest"), f(r, "mistakes: few positives"), f(r, "mistakes: fourth or lower")),
 "third wins 15-30% of near ties": lambda r: 0.15 <= f(r, "near ties: third wins") <= 0.30,
}
top = [r for r in oth if f(r, "accuracy") >= 0.945]
print(f"{'test':66} {'ARC':>4} {'of 37':>6} {'of top ' + str(len(top)):>9}  networks failing (of 37)")
for k, t in tests.items():
    res = [bool(t(r)) for r in oth]
    print(f"{k:66} {'yes' if t(arc) else 'no':>4} {sum(res):6d} {sum(bool(t(r)) for r in top):9d}  " + (" ".join(r['net'] for r, v in zip(oth, res) if not v) if sum(res) >= 19 else ""))
acc = np.array([f(r, "accuracy") for r in oth])
print("\nRank correlation with accuracy over the 37 (|rho| >= 0.4 shown):")
for k in arc:
    if k in ("net", "memory: ages", "accuracy"): continue
    v = np.array([f(r, k) for r in oth]); ok = ~np.isnan(v)
    if ok.sum() > 10 and np.nanstd(v) > 0:
        rho = spearmanr(acc[ok], v[ok]).correlation
        if abs(rho) >= 0.4: print(f"  {k:62} {rho:+.2f}")
