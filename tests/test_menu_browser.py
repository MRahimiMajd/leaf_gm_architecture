from playwright.sync_api import sync_playwright


def wait_for_menu_frame(page, timeout_s=30):
    """
    The menu is a third-party component and renders inside its own
    iframe. Wait until that iframe has appeared and return it.
    """
    waited = 0.0
    while waited < timeout_s:
        matches = [frame for frame in page.frames
                   if 'streamlit_option_menu' in frame.url]
        if matches:
            return matches[0]
        page.wait_for_timeout(500)
        waited += 0.5
    raise AssertionError('the menu iframe never appeared')


def test_real_menu_clicks_switch_pages(streamlit_server):
    """
    Click through the real menu add-on in a real (headless) browser.
    This is the only test that exercises the menu exactly like a
    visitor does - the test browser cannot click it (see the unmocked
    app test). Clicking 'Cross Validation' must show that page, and
    clicking 'About' must go back to the start page.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_server, wait_until='domcontentloaded',
                  timeout=60_000)

        menu = wait_for_menu_frame(page)
        entries = menu.locator('.nav-link')

        entries.filter(has_text='Cross Validation').first.click()
        page.get_by_role('heading', name='Cross Validation').wait_for(
            timeout=60_000)

        entries.filter(has_text='About').first.click()
        page.get_by_role('heading', name='Early Test Version').wait_for(
            timeout=60_000)

        browser.close()
