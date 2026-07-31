from playwright.sync_api import sync_playwright

def run_cuj(page):
    page.goto("http://localhost:8000")
    page.wait_for_timeout(1000)

    # Click memory button to open modal
    page.locator("#open-memory-btn").click()
    page.wait_for_timeout(1000)

    # Click the backdrop to dismiss
    page.mouse.click(10, 10) # Click near top left, outside modal shell
    page.wait_for_timeout(1000)

    # Open it again
    page.locator("#open-memory-btn").click()
    page.wait_for_timeout(1000)

    # Press Escape to dismiss
    page.keyboard.press("Escape")
    page.wait_for_timeout(1000)

    page.screenshot(path="/home/jules/verification/screenshots/verification.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
