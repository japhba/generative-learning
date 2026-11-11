/* Render toy_model_3_chain.html in a real browser: no page errors or external requests, no failed or leftover math, no unfilled
placeholders, no horizontal scroll, every figure loaded; save screenshots of each section. */
const { chromium } = require('/var/tmp/jbauer/vscode-jupyter-build/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/var/tmp/jbauer/generative_practice_chain/report_check';
const url = 'http://127.0.0.1:8095/toy_model_3_chain.html';

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
    assert.equal((await page.goto(url, { waitUntil: 'load' })).status(), 200);
    const state = await page.evaluate(() => ({
      katex: document.querySelectorAll('.katex').length, katexErrors: document.querySelectorAll('.katex-error').length,
      rawMath: /\\\(|\\\[|\$\$/.test(document.querySelector('main').innerText), placeholders: document.body.innerHTML.includes('{{'),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      images: [...document.querySelectorAll('figure img')].map(i => [i.getAttribute('src').split('/').pop(), i.naturalWidth, Math.round(i.getBoundingClientRect().width)]),
      wide: [...document.querySelectorAll('main *')].filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1).map(e => e.tagName + '.' + e.className).slice(0, 5),
      sections: [...document.querySelectorAll('section')].map(s => s.id), tables: document.querySelectorAll('table').length, words: document.querySelector('main').innerText.split(/\s+/).length,
    }));
    console.log(name, JSON.stringify(state));
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert(state.katex > 20); assert.equal(state.katexErrors, 0); assert(!state.rawMath); assert(!state.placeholders); assert(state.overflow <= 0, 'horizontal scroll');
    assert.equal(state.images.length, 10); for (const [src, w] of state.images) assert(w > 900, src);
    for (const id of state.sections) await page.locator('#' + id).screenshot({ path: `${out}/${name}-section-${id}.png` });
    await page.locator('header').screenshot({ path: `${out}/${name}-header.png` });
    await page.close();
  }
  await browser.close();
  console.log(out);
})();
