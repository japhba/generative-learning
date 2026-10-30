/* Render the practice-loop report in a real browser: no page errors, no failed or leftover math, no unfilled placeholders, no horizontal scroll; save screenshots. */
const { chromium } = require('/var/tmp/jbauer/vscode-jupyter-build/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/var/tmp/jbauer/generative_practice_loop_report';

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/var/tmp/jbauer/ms-playwright/chromium-1148/chrome-linux/chrome', headless: true, args: ['--no-sandbox'] });

  for (const [name, viewport, scheme] of [['desktop', { width: 1280, height: 900 }, 'light'], ['desktop-dark', { width: 1280, height: 900 }, 'dark'], ['phone', { width: 390, height: 844 }, 'light']]) {
    const page = await browser.newPage({ viewport, colorScheme: scheme });
    const errors = [], external = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('request', r => { if (!r.url().startsWith('http://127.0.0.1:8095/') && !r.url().startsWith('data:')) external.push(r.url()); });
    // On u435d the first navigation of a new tab in headless Chromium hangs (any page, even setContent); the next one loads in ms.
    await page.goto('http://127.0.0.1:8095/toy_model_1_ring.html', { timeout: 15000 }).catch(() => {});
    assert.equal((await page.goto('http://127.0.0.1:8095/toy_model_1_ring.html')).status(), 200);
    await page.waitForFunction(() => window.reportReady);
    const state = await page.evaluate(() => ({
      katex: document.querySelectorAll('.katex').length, katexErrors: document.querySelectorAll('.katex-error').length,
      rawMath: /\\\(|\\\[|\$\$/.test(document.querySelector('main').innerText), placeholders: document.body.innerHTML.includes('{{'),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      lines: [...document.querySelectorAll('#successChart polyline')].length,
    }));
    console.log(name, JSON.stringify(state));
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert(state.katex > 10); assert.equal(state.katexErrors, 0); assert(!state.rawMath); assert(!state.placeholders); assert(state.overflow <= 0);
    assert.equal(state.lines, 32 + 4);
    await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
    if (name === 'desktop') {
      for (const id of ['result', 'why', 'explore', 'checks']) await page.locator('#' + id).screenshot({ path: `${out}/${id}.png` });
      await page.locator('#successChart').scrollIntoViewIfNeeded();
      const box = await page.locator('#successChart').boundingBox();
      await page.mouse.move(box.x + box.width * .6, box.y + box.height * .5);
      assert((await page.locator('#successChart + .tooltip').textContent()).includes('Practice'));
      await page.locator('#successChart').locator('xpath=..').screenshot({ path: `${out}/hover.png` });
      await page.locator('#toStart').click();
      assert((await page.locator('#mapTitle').textContent()).startsWith('After 0 updates'));
      await page.locator('#explore').screenshot({ path: `${out}/explore-start.png` });
    }
    await page.close();
  }
  await browser.close();
  console.log(out);
})();
