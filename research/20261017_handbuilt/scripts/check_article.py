"""Test the article page (and v3/v4) of the browser model: scores match Python, no JS errors, every control works, screenshots."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
LIBS = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
page = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2] if len(sys.argv) > 2 else "viz/screens4"); out.mkdir(exist_ok=True)
m = json.load(open("viz/model.json"))
W_hi, W_hh, W_oh = (np.array(m[k]) for k in ("w_hi", "w_hh", "w_oh"))


def py_scores(x):
    h = np.zeros(16)
    for v in x:
        h = np.maximum(0, W_hh @ h + W_hi * v)
    s = W_oh @ h
    return s - s.mean()


def slide(pg, sel, v):
    pg.evaluate("([s, v]) => { const e = document.querySelector(s); e.value = v; e.dispatchEvent(new Event('input')); }", [sel, str(v)])


FIGS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": LIBS})
    for scheme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 900, "height": 900}, color_scheme=scheme)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda msg: errs.append(msg.text) if msg.type == "error" else None)
        pg.goto(page.as_uri())
        if scheme == "light":
            xs = np.round(np.random.default_rng(0).standard_normal((40, 10)), 2)
            js = np.array(pg.evaluate("xs => xs.map(r => scoresOf(run(r)[10]))", xs.tolist()))
            print("max |browser - python| score:", np.abs(js - np.array([py_scores(r) for r in xs])).max())
            for fid in FIGS:
                pg.locator(f"#{fid}").screenshot(path=str(out / f"{fid}_default.png"))
            before = [pg.inner_text(f"#{f}") for f in ("f1", "f2", "f9")]
            pg.click("#f0 [data-act=random]")
            print("shared list reaches sections 1, 2, playground:", [a != c for a, c in zip(before, [pg.inner_text(f"#{f}") for f in ("f1", "f2", "f9")])])
            pg.click("#f1 [data-act=prev]"); pg.click("#f1 [data-act=prev]")
            print("f1:", pg.inner_text("#f1 .cap"))
            pg.locator("#f3 .row rect").nth(5).click(); print("f3:", pg.inner_text("#f3 .cap"))
            for mode, want in (("none", "9"), ("big", "2"), ("wipe", "2"), ("unwipe", "1")):
                pg.click(f"#f4 [data-mode={mode}]"); print(f"f4 {mode} (Python says answer {want}):", pg.inner_text("#f4 .cap"))
            print("f10 dots:", pg.locator("#f10 circle").count())
            pg.select_option("#f9 [data-act=ex]", label="neuron 5 matters")
            pg.select_option("#f9 [data-act=model]", "hb"); print("f9 hand-built:", pg.inner_text("#f9 .result")); pg.select_option("#f9 [data-act=model]", "arc")
            slide(pg, "#f5 [data-act=v]", 90); slide(pg, "#f5 [data-act=s]", 120)
            print("f5:", pg.inner_text("#f5 .cap"))
            slide(pg, "#f6 [data-act=p1]", 6); slide(pg, "#f6 [data-act=p2]", 2)
            print("f6:", pg.inner_text("#f6 .cap"))
            print("f7:", pg.inner_text("#f7 .cap"))
            pg.locator("#f8 svg text.click", has_text="5").first.click()
            print("f8:", pg.inner_text("#f8 .cap"))
            pg.locator("#f9 .heat text.click", has_text="5").first.click()
            print("f9 after switching neuron 5 off:", pg.inner_text("#f9 .result"))
            print("f9 events:", pg.inner_text("#f9 .events")[:600])
            for fid in ("f3", "f4", "f5", "f6", "f8"):
                pg.locator(f"#{fid}").screenshot(path=str(out / f"{fid}_changed.png"))
        pg.screenshot(path=str(out / f"{scheme}_full.png"), full_page=True)
        print(scheme, "errors:", errs or "none")
        pg.close()
    b.close()
