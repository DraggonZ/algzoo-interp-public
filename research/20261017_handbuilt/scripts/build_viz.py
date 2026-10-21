"""Inline viz/model.json into viz/template.html -> viz/arc_2nd_argmax.html (self-contained)."""
import json
from pathlib import Path

v = Path("viz")
m = json.loads((v / "model.json").read_text())
import sys
src, dst = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("template.html", "arc_2nd_argmax.html")
html = (v / src).read_text().replace("/*MODEL*/null", json.dumps(m, separators=(",", ":")))
(v / dst).write_text(html)
print("wrote", v / dst, len(html) // 1024, "KB")
