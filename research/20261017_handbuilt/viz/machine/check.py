import json, os, sys
import numpy as np
from playwright.sync_api import sync_playwright
EXE = "/data/output/tools/chromium-154/chrome-headless-shell-linux64/chrome-headless-shell"
page = "file://" + os.path.abspath("second_largest_machine_v3.html")
d = json.load(open("page_data.json")); wi, wh, wo = (np.array(d[k]) for k in ("wi", "wh", "wo"))
def py(x):
    h = np.zeros(16)
    for v in x: h = np.maximum(0, wh @ h + wi * v)
    return wo @ h
libs = open("/data/output/tools/chromium-154/libpath.txt").read().strip()
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, env={**os.environ, "LD_LIBRARY_PATH": libs})
    for scheme, w, h in (("light", 1280, 900), ("dark", 1280, 900), ("light", 400, 800)):
        pg = b.new_page(viewport={"width": w, "height": h}, color_scheme=scheme)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(page); pg.wait_for_timeout(800)
        if w == 1280 and scheme == "light":
            xs = np.round(np.random.default_rng(0).standard_normal((20, 10)), 2)
            js = np.array(pg.evaluate("xs => xs.map(r => simulate(r, true).scores)", xs.tolist()))
            print("max |browser - python| score:", np.abs(js - np.array([py(r) for r in xs])).max())
            # Euler split check: sum of J x equals h at step 10
            err = pg.evaluate("""xs => { const R = simulate(xs, true); let e = 0; for (let j = 0; j < 16; j++) { let s = 0; for (let p = 0; p < 10; p++) s += R.Js[10][j][p] * xs[p]; e = Math.max(e, Math.abs(s - R.hs[10][j])); } return e; }""", xs[0].tolist())
            print("split error:", err)
            for s in range(0, 11):
                pg.evaluate(f"setStep({s})"); pg.wait_for_timeout(700 if s else 50)
                if s in (3, 6, 8, 10): pg.screenshot(path=f"shot_step{s}.png")
            pg.click("#iv-hide"); pg.wait_for_timeout(1500); print("stats with hide:", pg.inner_text("#stats").replace("\n", " | "))
            pg.click("#iv-hide"); pg.wait_for_timeout(800)
            pg.evaluate("window.scrollTo(0, 1400)"); pg.wait_for_timeout(300); pg.screenshot(path="shot_scrolled.png")
            print("page width:", pg.evaluate("document.documentElement.scrollWidth"))
        else:
            pg.evaluate("setStep(6)"); pg.wait_for_timeout(900)
            pg.screenshot(path=f"shot_{scheme}_{w}.png", full_page=(w == 400))
            print(scheme, w, "page width:", pg.evaluate("document.documentElement.scrollWidth"))
        print(scheme, w, "errors:", errs)
    b.close()
