"""Run the frozen v2c scorer (unchanged) on a null copy of ARC.

Usage:  python -u score_null.py KIND SEED path/to/rule.py      (KIND: arc, block, row)

The null's weights replace ARC's everywhere the scorer and the rule read them (algzoo.core.example); ARC's Fisher stays
the precision metric (the scorer loads it from file)."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nulls  # noqa: E402

SCORER = Path(__file__).resolve().parents[1] / "memory_register" / "score_memory_rule_v2c.py"

if __name__ == "__main__":
    kind, seed, rule = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    nulls.install(kind, seed)
    print(f"null {kind} seed {seed}", flush=True)
    sys.argv = [str(SCORER), rule]
    runpy.run_path(str(SCORER), run_name="__main__")
