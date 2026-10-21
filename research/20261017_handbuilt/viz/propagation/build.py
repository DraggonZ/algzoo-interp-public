"""Build propagation.html from template.html and data.json (run data.py first)."""
from pathlib import Path
here = Path(__file__).parent
html = (here / "template.html").read_text().replace("/*DATA*/", (here / "data.json").read_text())
(here.parent / "propagation.html").write_text(html); print("wrote", here.parent / "propagation.html", len(html))
