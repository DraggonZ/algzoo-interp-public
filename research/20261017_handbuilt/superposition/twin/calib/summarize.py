"""Collapse the nudge battery and lead shifts into separate questions, and place each network against two references:
ARC with small weight noise (the same mechanism, slightly perturbed) and the other trained 16-neuron networks (same
recipe, different solutions). One number per question per network:
  memory nudge   lead shift from memory x1.05 at step 5, mean of the two list types (units of the second's vote)
  store nudge    lead shift from store x1.01 at step 5, mean of the two list types
  keep nudge     lead shift from keep gate x1.05 at every step, last-step contender, mean of the two list types
  hiding nudge   lead shift of largest over second from hiding x0.9 at step 5
  near-tie order how much better the earlier contender does when it is truly second than when it is third, points
                 (mean over the three list types of 'earlier wins'): 100 means near ties are decided perfectly
  late bias      step-9 contender wins when truly second half the time, percent
  hiding alone   largest named with the hiding neurons silenced at every step, percent
  shift register mean of the extra-step and appended-number tests, percent as intended
  sensitivity    correlation of log answer changes over the 40 group-by-step cells with ARC's
Distance to ARC per question is divided by the median distance of the trained siblings, so 1 means 'as far as a
typical other trained network'."""
import numpy as np
def questions(r):
    b, s = r["battery"], r["shifts"]
    q = {}
    q["memory nudge"] = np.mean([s["memory x1.05 step 5, earlier minus later (2nd early)"], s["memory x1.05 step 5, earlier minus later (3rd early)"]])
    q["store nudge"] = np.mean([s["store x1.01 step 5, earlier minus later (2nd early)"], s["store x1.01 step 5, earlier minus later (3rd early)"]])
    q["keep nudge"] = np.mean([s["keep x1.05 all steps, step-9 contender minus other (2nd at 9)"], s["keep x1.05 all steps, step-9 contender minus other (3rd at 9)"]])
    q["hiding nudge"] = s["hiding x0.9 step 5, largest minus second"]
    pairs = [("near tie, step 5, 2nd early: earlier wins", "near tie, step 5, 3rd early: earlier wins"),
             ("near tie, step 7, 2nd early: earlier wins", "near tie, step 7, 3rd early: earlier wins"),
             ("near tie, largest and 2nd early by step 3, other after 5: earlier wins", "near tie, largest and 3rd early by step 3, other after 5: earlier wins")]
    q["near-tie order"] = np.mean([b[a] - b[c] for a, c in pairs])
    q["late bias"] = b["late bias: step-9 contender wins (truly second half the time)"]
    q["hiding alone"] = b["  hiding silenced at all steps (all lists)"]
    q["shift register"] = np.mean([b[k] for k in ("extra input 0: names second-1", "append 2 x largest: names largest-1", "append between largest and second: names 9", "append -3: names second-1")])
    q["_sens"] = b["_sensitivity"]; q["accuracy"] = b["accuracy"]
    return q
if __name__ == "__main__":
    res = {k: v for k, v in np.load("calib.npy", allow_pickle=True).item().items() if v is not None}
    Q = {k: questions(v) for k, v in res.items()}
    A = Q["ARC"]; la = np.log10(A["_sens"] + 0.01)
    for k in Q: Q[k]["sensitivity"] = np.corrcoef(la, np.log10(Q[k]["_sens"] + 0.01))[0, 1]
    names = ["memory nudge", "store nudge", "keep nudge", "hiding nudge", "near-tie order", "late bias", "hiding alone", "shift register", "sensitivity"]
    sib = [k for k in Q if k.startswith("sibling")]; noi = [k for k in Q if k.startswith("ARC noise")]; tw = [k for k in Q if k not in sib + noi + ["ARC"]]
    scale = {n: np.median([abs(Q[k][n] - A[n]) for k in sib]) for n in names}
    print(f"{len(sib)} trained siblings with complete groups; accuracy {min(Q[k]['accuracy'] for k in sib):.1f}-{max(Q[k]['accuracy'] for k in sib):.1f}%\n")
    print(f"{'question':16s}{'ARC':>9s}{'siblings: median (10-90%)':>30s}{'noisy ARC 0.001/0.003/0.01':>30s}" + "".join(f"{k[:14]:>16s}" for k in tw))
    for n in names:
        sv = np.array([Q[k][n] for k in sib]); nv = [np.mean([Q[k][n] for k in noi if f"{sd:g} #" in k]) for sd in (0.001, 0.003, 0.01)]
        print(f"{n:16s}{A[n]:+9.3f}{np.median(sv):+12.3f} ({np.quantile(sv, .1):+.3f} to {np.quantile(sv, .9):+.3f})   {nv[0]:+8.3f}{nv[1]:+8.3f}{nv[2]:+8.3f}  " + "".join(f"{Q[k][n]:+16.3f}" for k in tw))
    print(f"\n{'distance to ARC / typical sibling distance':44s}" + "".join(f"{k[:14]:>16s}" for k in tw + ['noisy 0.003'] + ['median sibling']))
    for n in names:
        nd = np.mean([abs(Q[k][n] - A[n]) for k in noi if "0.003 #" in k]) / scale[n]
        print(f"{n:44s}" + "".join(f"{abs(Q[k][n] - A[n]) / scale[n]:16.2f}" for k in tw) + f"{nd:16.2f}{1.0:16.2f}")
    tot = {k: np.median([abs(Q[k][n] - A[n]) / scale[n] for n in names]) for k in tw + sib + noi}
    print("\nmedian over questions of distance / typical sibling distance:")
    for k in tw + noi: print(f"   {k:30s} {tot[k]:.2f}   (accuracy {Q[k]['accuracy']:.1f}%)")
    sv = sorted(tot[k] for k in sib); print(f"   siblings: median {np.median(sv):.2f}, closest {sv[0]:.2f}, 10-90% {np.quantile(sv, .1):.2f}-{np.quantile(sv, .9):.2f}")
