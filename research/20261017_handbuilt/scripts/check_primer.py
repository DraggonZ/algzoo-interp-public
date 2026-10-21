"""Check viz/primer.html: no JS errors, JS network matches Python (trained and hand-built), screenshots of each figure at chosen steps."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
LIBS = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
root = Path(__file__).resolve().parent.parent
page = root / "viz/primer.html"
out = root / "viz/screens_primer"
out.mkdir(exist_ok=True)
m = json.load(open(root / "viz/model.json"))
Wi, Wh, Wo = (np.array(m[k]) for k in ("w_hi", "w_hh", "w_oh"))


def py_scores(x):
    h = np.zeros(16)
    for v in x:
        h = np.maximum(0, Wh @ h + Wi * v)
    return Wo @ h


# figure id -> list of (label, number of clicks on "next" from the first step, extra JS before)
SHOTS = {
    "f1": [("a", 0, "document.querySelector('#f1 [data-a=show]').click()")],
    "f2": [("end", None, None), ("s4", 4, None)],
    "f3": [("s5", 5, None), ("end", 10, None), ("rule0", 10, "const s=document.querySelector('#f3 [data-a=rule]'); s.value='0'; s.onchange({target:s})")],
    "f4": [("s7", 7, None)],
    "f5": [("toy", 10, None), ("real", 10, "const s=document.querySelector('#f5 [data-a=mode]'); s.value='real'; s.onchange()")],
    "f6": [("one", None, None), ("all", None, "['d2','d3','d4'].forEach(a=>document.querySelector('#f6 [data-a='+a+']').click())")],
    "f6b": [("s5", 5, None), ("end", 10, None)],
    "f8b": [("t", None, None)],
    "f7": [("s0", 0, None), ("s2", 2, None), ("s3", 3, None)],
    "f8": [("s4", 4, None), ("s7", 7, None), ("end", 10, None)],
    "f9": [("s7", 7, None), ("end", 10, None)],
    "f10": [("s5", 5, None), ("end", 10, None)],
    "f11": [("end", None, None), ("s5", 5, None)],
    "f12": [("t", None, None)],
    "f13": [("t", None, None)],
    "f14": [("t", None, None)],
}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": LIBS})
    pg = b.new_page(viewport={"width": 820, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda msg: errs.append(msg.text) if msg.type == "error" else None)
    pg.emulate_media(reduced_motion="reduce")
    pg.goto(page.as_uri())
    pg.wait_for_timeout(1500)
    xs = np.round(np.random.default_rng(0).standard_normal((30, 10)), 2)
    js = pg.evaluate("xs => xs.map(x => readout(M, run(M, x).H[9]))", xs.tolist())
    print("trained max |js - py|:", max(np.abs(np.array(j) - py_scores(x)).max() for j, x in zip(js, xs)))
    acc = pg.evaluate("""() => { seed = 5; let ok = 0, okh = 0; const n = 20000; for (let j = 0; j < n; j++) { const l = Array.from({length: 10}, randn);
        const t = order(l)[1]; if (answerOf(M, l) === t) ok++; if (answerOf(HB, l) === t) okh++; } return [ok / n, okh / n]; }""")
    print("accuracy in browser, trained and hand-built:", acc)
    print("cartoon:", pg.inner_text("#acc-cartoon"))
    for fid, shots in SHOTS.items():
        for lab, n, pre in shots:
            if pre:
                pg.evaluate(pre)
            if n is not None:
                if pg.query_selector(f"#{fid} [data-a=first]"):
                    pg.click(f"#{fid} [data-a=first]")
                for _ in range(n):
                    pg.click(f"#{fid} [data-a=next]")
            pg.wait_for_timeout(150)
            el = pg.query_selector(f"#{fid}")
            el.screenshot(path=str(out / f"{fid}_{lab}.png"))
            print(fid, lab, "|", pg.inner_text(f"#{fid} .cap")[:400].replace("\n", " "))
    print("errors:", errs)
    b.close()
