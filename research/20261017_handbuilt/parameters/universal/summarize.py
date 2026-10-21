"""ARC against the 37 for every quantity in quantities.csv: ARC's value, the 37's median and 10-90% range, how many of
the 37 have the quantity at all, and ARC's percentile (share of the 37 below ARC). Writes summary.log."""
import csv, numpy as np
rows = list(csv.DictReader(open("quantities.csv"))); arc = rows[0]; oth = rows[1:]
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan
print(f"{'quantity':62} {'ARC':>8} {'median':>8} {'10%':>8} {'90%':>8} {'have':>5} {'ARC pct':>7}")
for k in arc:
    if k in ("net", "memory: ages"): continue
    a = num(arc[k]); v = np.array([num(r[k]) for r in oth]); ok = ~np.isnan(v)
    pct = np.mean(v[ok] < a) * 100 if ok.any() and not np.isnan(a) else np.nan
    print(f"{k:62} {a:8.3f} {np.median(v[ok]) if ok.any() else np.nan:8.3f} {np.percentile(v[ok], 10) if ok.any() else np.nan:8.3f} "
          f"{np.percentile(v[ok], 90) if ok.any() else np.nan:8.3f} {ok.sum():5d} {pct:7.0f}")
print("\nmemory switch-off ages:", "ARC", arc["memory: ages"], "|", "; ".join(f"{r['net']} {r['memory: ages']}" for r in oth))
