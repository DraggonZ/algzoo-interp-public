"""Test the browser model in headless Chromium: scores match Python, no JS errors, controls work,
screenshots of each figure (light and dark). Run with the playwright env:
/data/output/tools/playwright-env/bin/python scripts/check_viz.py viz/arc_2nd_argmax_v2.html"""
import json
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
page_path = Path(sys.argv[1]).resolve()
out = Path("viz/screens"); out.mkdir(exist_ok=True)
m = json.load(open("viz/model.json"))
W_hi, W_hh, W_oh = (np.array(m[k]) for k in ("w_hi", "w_hh", "w_oh"))


def py_scores(x):
    h = np.zeros(16)
    for v in x:
        h = np.maximum(0, W_hh @ h + W_hi * v)
    s = W_oh @ h
    return s - s.mean()


with sync_playwright() as p:
    import os
    libs = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": libs})
    for scheme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 960, "height": 900}, color_scheme=scheme)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda msg: errs.append(msg.text) if msg.type == "error" else None)
        pg.goto(page_path.as_uri())
        if scheme == "light":
            xs = np.round(np.random.default_rng(0).standard_normal((40, 10)), 2)
            js = np.array(pg.evaluate("xs => xs.map(r => scoresOf(run(r)[10]))", xs.tolist()))
            print("max |browser - python| score:", np.abs(js - np.array([py_scores(r) for r in xs])).max())
            # exercise controls
            pg.click("#f0 [data-act=random]"); pg.click("#f1 [data-act=random]"); pg.click("#f5 [data-act=random]")
            pg.fill("#f2 [data-act=pos]", "7"); pg.dispatch_event("#f2 [data-act=pos]", "input")
            pg.fill("#f4 [data-act=v]", "95"); pg.dispatch_event("#f4 [data-act=v]", "input")
            pg.select_option("#f6 [data-act=ex]", "close call, right")
            pg.click("#f7 [data-act=prev]"); pg.click("#f7 [data-act=next]")
            pg.click("#f7 .heat text:text('3')")
            print("f2 caption:", pg.inner_text("#f2 .cap"))
            print("f3 caption:", pg.inner_text("#f3 .cap"))
            print("f4 caption:", pg.inner_text("#f4 .cap"))
            print("playground result:", pg.inner_text("#f7 .result"))
            # shared input: a new random input in the first figure must reach figures 1, 5 and the playground
            before = [pg.inner_text(f"#{f}") for f in ("f1", "f5", "f7")]
            pg.click("#f0 [data-act=random]")
            after = [pg.inner_text(f"#{f}") for f in ("f1", "f5", "f7")]
            print("shared input updates f1, f5, playground:", [a != b for a, b in zip(before, after)])
            pg.select_option("#f7 [data-act=ex]", "typical")
            print("example picked in the playground reaches the first figure:", "typical" and pg.inner_text("#f0 .result"))
            pg.click("#f8 svg text:text('5')")
        for fid in ("f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"):
            pg.locator(f"#{fid}").screenshot(path=str(out / f"{scheme}_{fid}.png"))
        pg.screenshot(path=str(out / f"{scheme}_full.png"), full_page=True)
        print(scheme, "errors:", errs or "none")
        pg.close()
    b.close()
