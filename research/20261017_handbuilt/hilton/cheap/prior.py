"""Copies the earlier results this README compares with, verbatim, from propagate/ (so every number cited here is in a
log of this folder)."""
import common as c
log = c.Log("prior.log")
P = "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/"
for f in ("full_mean.log", "converge.log", "cheap_interp.log", "bands_across.log", "where_wrong.log"):
    log(f"== propagate/{f}")
    for line in open(P + f): log("   " + line.rstrip())
lines = open(P + "across.log").read().splitlines(); log("== propagate/across.log (last two lines)"); [log("   " + l) for l in lines[-2:]]
