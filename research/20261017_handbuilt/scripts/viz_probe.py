"""Open the interactive page in a headless browser, optionally run JavaScript (to click or change
something), then save a screenshot and print the visible text.

Usage:
  /data/output/tools/playwright-env/bin/python scripts/viz_probe.py OUT.png [SELECTOR] [JS]
  SELECTOR: CSS selector of the part to screenshot (default: whole page), e.g. "#f2"
  JS: JavaScript to run first, e.g. "document.querySelector('#f2 input').value=5; document.querySelector('#f2 input').dispatchEvent(new Event('input'))"
Figures have ids f0, f1, f2, f3, f3b, f4, f5, f6, f7, f8, f9 in page order. The text printed is the visible text of the screenshotted part."""
import os
import sys

from playwright.sync_api import sync_playwright

PAGE = "file:///data/output/research/algzoo/20261017_handbuilt/viz/arc_2nd_argmax_v3.html"
EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
out = sys.argv[1]
sel = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
js = sys.argv[3] if len(sys.argv) > 3 else None
libs = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": libs})
    pg = b.new_page(viewport={"width": 960, "height": 1000})
    pg.goto(os.environ.get("VIZ_PAGE", PAGE))
    if js:
        pg.evaluate(js)
    if sel:
        pg.locator(sel).screenshot(path=out)
        print(pg.inner_text(sel))
    else:
        pg.screenshot(path=out, full_page=True)
        print(pg.inner_text("body"))
    b.close()
