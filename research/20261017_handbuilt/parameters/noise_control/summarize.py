"""Tables from rules_*.jsonl, searches_*.jsonl, mechanisms_*.jsonl -> printed markdown (summary.log)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    p = HERE / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def stats(v):
    v = np.array(v, float)
    return f"{v.mean():+.0f} ± {v.std():.0f}, max {v.max():+.0f}" if len(v) else "—"


print("## Step 3: rules of fixed form (net bits, v2c)\n")
print("ARC: as the frozen scorer gives it. Nulls: mean ± spread, maximum over 20 copies. 'best of' = the most generous "
      "of own fit, exact fit and row fallback per copy. 'gross' = row fallback before paying for structure and spreads.\n")
print("| round | rule | ARC | block nulls, own fit | block nulls, best of | row nulls, best of | ARC gross | block nulls gross, max | row nulls gross, max |")
print("|---|---|---|---|---|---|---|---|---|")
arc = {r["round"]: r for r in read("rules_arc.jsonl")}
blk, row = read("rules_block.jsonl"), read("rules_row.jsonl")
for rnd, a in sorted(arc.items()):
    b = [r for r in blk if r["round"] == rnd]; w = [r for r in row if r["round"] == rnd]
    best = lambda rs: [max(r["own"], r["exact"], r["fallback"]) for r in rs]
    gmax = lambda rs: f"{max(r['gross'] for r in rs):+.0f}" if rs else "—"
    print(f"| {rnd} | `{a['rule']}` | {a['own']:+.1f} | {stats([r['own'] for r in b])} | {stats(best(b))} | "
          f"{stats(best(w))} | {a['gross']:+.0f} | {gmax(b)} | {gmax(w)} |")
sing = sum(1 for r in blk + row if r.get("singular"))
print(f"\nRule evaluations on nulls that hit a singular state matrix (dead neuron; minimum-norm fallback): {sing} of {len(blk + row)}.")

print("\n## Step 4: searches rerun (gain of the best candidate over the starting rule, bits, before any charge)\n")
charge = {"r8": "11", "r9": "3 (audit 7)", "r14": "5", "r15": "5"}
cnum = {"r8": 11, "r9": 3, "r14": 5, "r15": 5}
pick = {"r8": "(3, -1, -1, -1)", "r9": "rows (7,) hiding", "r14": "= min", "r15": "top 3"}
rq = read("searches_resid.jsonl")
print("Main test, residual nulls (`run_searches_resid.py`; constant held at ARC's fit for rounds 8-9, fitted to the bits"
      " for rounds 14-15; ARC's row uses the same fit). A search may also keep nothing, so a negative best counts as 0"
      " in the 'above the charge' column.\n")
print("| search | candidates | charge applied | ARC: best pick, gain | ARC: the night's pick, gain | nulls: best gain, median | 95th percentile | max | nulls above the charge |")
print("|---|---|---|---|---|---|---|---|---|")
for s in ("r8", "r9", "r14", "r15"):
    a = [r for r in rq if r["search"] == s and r["seed"] < 0][0]
    g = np.array([r["gain"] for r in rq if r["search"] == s and r["seed"] >= 0])
    print(f"| round {s[1:]} | {a['n']} | {charge[s]} | {a['best']}, {a['gain']:+.1f} | {pick[s]}, {a['all'][pick[s]] - a['base']:+.1f} | "
          f"{np.median(g):+.1f} | {np.percentile(g, 95):+.1f} | {g.max():+.1f} | {int((g > cnum[s]).sum())} of {len(g)} |")
print("\nFor comparison, block and row nulls (`run_searches.py`; constant at the exact minimum of the default objective)."
      " The starting rule is already broken on these copies (step 3), so a search 'gains' by repairing it; not used for"
      " conclusions.\n")
print("| search | ARC: best pick, gain | block nulls: mean ± spread, max | row nulls: mean ± spread, max |")
print("|---|---|---|---|")
sa = {r["search"]: r for r in read("searches_arc.jsonl")}
sb, sr = read("searches_block.jsonl"), read("searches_row.jsonl")
for s, a in sa.items():
    b = [r["gain"] for r in sb if r["search"] == s]; w = [r["gain"] for r in sr if r["search"] == s]
    print(f"| round {s[1:]} | {a['best']}, {a['gain']:+.1f} | {stats(b)} | {stats(w)} |")
print("\n## Step 5: the best rule by mechanism (net bits)\n")
ma = read("mechanisms_arc.jsonl"); mb, mr = read("mechanisms_block.jsonl"), read("mechanisms_row.jsonl")
if ma:
    a = ma[0]
    print("| mechanism | ARC alone | block nulls alone | row nulls alone | ARC in the full rule (drop) | block nulls drop | row nulls drop |")
    print("|---|---|---|---|---|---|---|")
    for m in ["push", "zero_sums", "hiding", "detector", "delay"]:
        print(f"| {m} | {a[m + '_alone']:+.1f} | {stats([r[m + '_alone'] for r in mb])} | {stats([r[m + '_alone'] for r in mr])} | "
              f"{a[m + '_drop']:+.1f} | {stats([r[m + '_drop'] for r in mb])} | {stats([r[m + '_drop'] for r in mr])} |")
    print(f"| contender cut (r15a − r14a) | — | — | — | {a['cut']:+.1f} | {stats([r['cut'] for r in mb])} | {stats([r['cut'] for r in mr])} |")
    print(f"| full rule | {a['full']:+.1f} | {stats([r['full'] for r in mb])} | {stats([r['full'] for r in mr])} | | | |")
