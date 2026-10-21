"""Build viz/primer.html: inject the trained weights, hand-built v5 weights, dial data, examples and estimator rows."""
import json
from pathlib import Path

from algzoo.handbuilt import build_v5

viz = Path(__file__).resolve().parent.parent / "viz"
m = json.loads((viz / "model.json").read_text())
net = build_v5()
wi, wh, wo = net.params
data = {
    "TRAINED": {k: m[k] for k in ("w_hi", "w_hh", "w_oh")},
    "HB": {"w_hi": wi.tolist(), "w_hh": wh.tolist(), "w_oh": wo.tolist()},
    "DIAL": {k: m[k] for k in ("dial_neurons", "dial_mean", "dial_axes", "spokes")},
    "EX": m["examples"],
    "EST": m["estimator"],
}
html = (viz / "primer_template.html").read_text()
for k, v in data.items():
    html = html.replace(f"/*{k}*/null", json.dumps(v, separators=(",", ":")))
(viz / "primer.html").write_text(html)
print("wrote", viz / "primer.html", len(html))
