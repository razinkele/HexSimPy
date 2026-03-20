"""Playwright test: screenshot deck.gl hex renderings for visual comparison."""
import asyncio
from pathlib import Path


async def screenshot_html_files():
    from playwright.async_api import async_playwright

    output_dir = Path("tests/hex_test_output")
    html_files = sorted(output_dir.glob("*_deckgl.html"))

    if not html_files:
        print("No HTML files found!")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 800, "height": 600})

        for html_file in html_files:
            name = html_file.stem.replace("_deckgl", "")
            url = f"file:///{html_file.resolve().as_posix()}"
            print(f"Loading: {name}...")

            await page.goto(url)
            # Wait for deck.gl to render
            await page.wait_for_timeout(3000)

            screenshot_path = output_dir / f"{name}_deckgl_screenshot.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"  Screenshot: {screenshot_path}")

        await browser.close()

    print("\nDone! Compare *_reference.png with *_deckgl_screenshot.png")


if __name__ == "__main__":
    asyncio.run(screenshot_html_files())
