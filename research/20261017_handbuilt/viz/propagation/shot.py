import os, sys
EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
LIBS = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": LIBS}); pg = b.new_page(viewport={"width": 800, "height": 1000}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto("file:///data/output/research/algzoo/20261017_handbuilt/viz/propagation.html")
    pg.screenshot(path=sys.argv[1] + "_a.png", full_page=True)
    pg.click("#bFade"); pg.click("#bShuf"); pg.fill("#tS", "8"); pg.dispatch_event("#tS", "input")
    pg.screenshot(path=sys.argv[1] + "_b.png", full_page=True)
    pg.set_viewport_size({"width": 390, "height": 900}); w = pg.evaluate("document.documentElement.scrollWidth")
    print("errors", errs, "phone scroll width", w); b.close()
