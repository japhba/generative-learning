/* Render toy_model_2_WS.html in a real browser: no page errors or external requests, no failed or leftover math, no unfilled
placeholders, no horizontal scroll; run the live simulation and the explorer; save screenshots. */
const { chromium } = require('/var/tmp/jbauer/vscode-jupyter-build/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/var/tmp/jbauer/generative_practice_disentangle/layers/report_check';
const url = 'http://127.0.0.1:8095/toy_model_2_WS.html';

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/var/tmp/jbauer/ms-playwright/chromium-1148/chrome-linux/chrome', headless: true, args: ['--no-sandbox'] });
  for (const [name, viewport, scheme] of [['desktop', { width: 1280, height: 900 }, 'light'], ['desktop-dark', { width: 1280, height: 900 }, 'dark'], ['phone', { width: 390, height: 844 }, 'light']]) {
    const page = await browser.newPage({ viewport, colorScheme: scheme });
    const errors = [], external = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('response', r => { if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) errors.push(`${r.status()} ${r.url()}`); });
    page.on('request', r => { if (!r.url().startsWith('http://127.0.0.1:8095/') && !r.url().startsWith('data:')) external.push(r.url()); });
    // On u435d the first navigation of a new tab in headless Chromium hangs (any page, even setContent); the next one loads in ms.
    await page.goto(url, { timeout: 15000 }).catch(() => {});
    assert.equal((await page.goto(url)).status(), 200);
    await page.waitForFunction(() => window.reportReady && window.explorerReady);
    const state = await page.evaluate(() => ({
      katex: document.querySelectorAll('.katex').length, katexErrors: document.querySelectorAll('.katex-error').length,
      rawMath: /\\\(|\\\[|\$\$/.test(document.querySelector('main').innerText), placeholders: document.body.innerHTML.includes('{{'),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      genLines: document.querySelectorAll('#genChart polyline').length, cells: document.querySelectorAll('#hmG rect').length,
    }));
    console.log(name, JSON.stringify(state));
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert(state.katex > 30); assert.equal(state.katexErrors, 0); assert(!state.rawMath); assert(!state.placeholders); assert(state.overflow <= 0);
    assert.equal(state.genLines, 4); assert.equal(state.cells, 36);
    if (name === 'desktop') {  // run the live simulation to the end at 12x and read the final angles
      await page.selectOption('#simSpeed', '12'); await page.click('#simPlay');
      await page.waitForFunction(() => /update 3,000/.test(document.getElementById('readB').textContent), null, { timeout: 60000 });
      console.log('sim', await page.evaluate(() => [document.getElementById('readB').innerText, document.getElementById('readG').innerText]));
      await page.locator('#rotation').screenshot({ path: `${out}/sim-end.png` });
      await page.click('#exStart'); await page.locator('#align').screenshot({ path: `${out}/explorer-start.png` });
      await page.click('#exEnd'); await page.locator('#align').screenshot({ path: `${out}/explorer-end.png` });
      await page.selectOption('#exLearner', 'gaussian'); await page.locator('.heatgrid').screenshot({ path: `${out}/explorer-gaussian.png` });
      await page.selectOption('#exLearner', 'two_layer');
      for (const id of ['summary', 'model', 'prior', 'twin', 'spec', 'meaning', 'rotation']) await page.locator('#' + id).screenshot({ path: `${out}/section-${id}.png` });
      await page.locator('header').screenshot({ path: `${out}/section-header.png` });
    }
    await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
    await page.close();
  }
  await browser.close();
  console.log(out);
})();
