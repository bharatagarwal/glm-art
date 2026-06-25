import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        p = await b.new_page(viewport={"width": 1280, "height": 800})
        errs = []
        p.on("pageerror", lambda e: errs.append(f"[pageerror] {e}"))
        await p.goto("file:///Users/bharat/repos/glm-art/output/fluviglyph.html",
                     wait_until="networkidle", timeout=60000)
        await p.wait_for_timeout(3000)
        await p.screenshot(path="output/shot_0_initial.png")
        # hide loading if still there, then play
        await p.evaluate("document.getElementById('loading').style.display='none'")
        await p.click("#play")
        await p.wait_for_timeout(4000)
        await p.screenshot(path="output/shot_1_mid.png")
        rep = await p.evaluate("""()=>({frac:window.__fluviglyph?.frac, stone:window.__fluviglyph?.stoneMat?.uniforms?.uOpacity?.value, blend:window.__fluviglyph?.stoneMat?.uniforms?.uBlend?.value, artifact:!!(window.__fluviglyph?.artifacts?.visible)})""")
        print("mid report:", rep)
        await p.fill("#scrub", "850")
        await p.wait_for_timeout(1500)
        await p.screenshot(path="output/shot_2_iter170.png")
        await p.fill("#scrub", "980")
        await p.wait_for_timeout(2000)
        await p.screenshot(path="output/shot_3_end.png")
        await b.close()
        print("errors:", errs or "(none)")

asyncio.run(main())
