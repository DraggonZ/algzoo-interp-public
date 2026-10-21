"""r5a: quick-evaluator sweep of rules/backpush.py allocations (block rows S, equations 1..m), gain held at FIXG.
Usage: python r5a_sweep.py "14:1" "14,15:2" ...   Each config runs in its own process (the rule reads env at load)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cfg in sys.argv[1:]:
    rows, m = cfg.split(":")
    env = dict(os.environ, BP_ROWS=rows, BP_M=m, FIXG=os.environ.get("FIXG", "0.92"))
    out = subprocess.run([sys.executable, "-u", str(HERE / "r4b_quick.py"), str(HERE / "rules" / "backpush.py")],
                         env=env, capture_output=True, text=True).stdout
    print(f"== S={rows} m={m}\n{out.strip()}", flush=True)
