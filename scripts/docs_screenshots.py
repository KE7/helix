"""Regenerate the `helix ui` screenshots that the docs show.

Point it at a dashboard serving a real run, during or after `helix evolve`::

    helix ui --no-browser        # prints http://127.0.0.1:8790/?token=...
    uv run --with playwright python scripts/docs_screenshots.py "http://127.0.0.1:8790/?token=..."

It writes, in light and dark, to ``docs/assets/ui/``:

- ``helix-ui-{light,dark}.png``: the whole window on the Loop, as the run stands;
- ``lineage-{light,dark}.png``: the Lineage view, cropped to the view;
- ``path-{light,dark}.png``: the Path view of the best candidate, cropped to the view.

It drives the installed Google Chrome. ``--chromium`` uses Playwright's own browser instead (run
``uv run --with playwright playwright install chromium`` once). Nothing is written to the run.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import Page, async_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "ui"
VIEWPORT = {"width": 1440, "height": 900}
# name, route, the element to crop to (None for the whole window), device scale factor
SHOTS = [
    ("helix-ui", "", None, 1),
    ("lineage", "#lineage", "#v-lineage", 2),
    ("path", "#path", "#v-path", 2),
]


async def _settle(page: Page, route: str) -> None:
    await page.evaluate(f"location.hash = {route!r}")
    # the views fade in over 0.45 s and load their data on first open
    await page.wait_for_timeout(1500)
    if route == "#lineage":
        await page.wait_for_selector("#lg .ln circle")
    if route == "#path":
        await page.wait_for_selector("#v-path .pst")
    await page.mouse.move(2, 2)  # no hover card or lit path in the picture
    await page.wait_for_timeout(300)


async def main(url: str, out: Path, chromium: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await (p.chromium.launch() if chromium else p.chromium.launch(channel="chrome"))
        for scheme in ("light", "dark"):
            for name, route, crop, scale in SHOTS:
                ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=scale, color_scheme=scheme)
                page = await ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                await page.goto(url)
                await page.wait_for_selector("#v-map svg *")
                await _settle(page, route)
                path = out / f"{name}-{scheme}.png"
                if crop:
                    box = await page.locator(crop).bounding_box()
                    assert box, f"{crop} is not on the page"
                    await page.screenshot(path=path, clip=box)
                else:
                    await page.screenshot(path=path)
                print(f"{path}  {path.stat().st_size // 1024} KB" + (f"  page errors: {errors}" if errors else ""))
                await ctx.close()
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="the address `helix ui` prints, token included")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output directory (default: {OUT})")
    ap.add_argument("--chromium", action="store_true", help="use Playwright's Chromium instead of Google Chrome")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.out, a.chromium))
