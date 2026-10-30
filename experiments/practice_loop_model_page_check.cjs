/* Check the model page: the JS learner reproduces the saved run (exact success and loss at saved checkpoints;
   mean final success over 32 worlds within sampling error), plus rendering checks and screenshots. */
const { chromium } = require('/var/tmp/jbauer/vscode-jupyter-build/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/var/tmp/jbauer/generative_practice_loop_report/model';
const URL = 'http://127.0.0.1:8095/practice-loop-model.html';

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/var/tmp/jbauer/ms-playwright/chromium-1148/chrome-linux/chrome', headless: true, args: ['--no-sandbox'] });

  for (const [name, viewport, scheme] of [['desktop', { width: 1280, height: 900 }, 'light'], ['desktop-dark', { width: 1280, height: 900 }, 'dark'], ['phone', { width: 390, height: 844 }, 'light']]) {
    const page = await browser.newPage({ viewport, colorScheme: scheme });
    const errors = [], external = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('request', r => { if (!r.url().startsWith('http://127.0.0.1:8095/') && !r.url().startsWith('data:')) external.push(r.url()); });
    // On u435d the first navigation of a new tab in headless Chromium hangs (any page, even setContent); the next one loads in ms.
    await page.goto(URL, { timeout: 15000 }).catch(() => {});
    assert.equal((await page.goto(URL)).status(), 200);
    await page.waitForFunction(() => window.reportReady);
    const state = await page.evaluate(() => ({
      katex: document.querySelectorAll('.katex').length, katexErrors: document.querySelectorAll('.katex-error').length,
      rawMath: /\\\(|\\\[|\$\$/.test(document.querySelector('main').innerText), placeholders: document.body.innerHTML.includes('{{'),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));
    console.log(name, JSON.stringify(state));
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    assert(state.katex > 40); assert.equal(state.katexErrors, 0); assert(!state.rawMath); assert(!state.placeholders); assert(state.overflow <= 0);
    if (name === 'desktop') {
      const fidelity = await page.evaluate(() => {
        let success = 0, internal = 0, loss = 0;
        for (const W of D.worlds) W.check.frames.forEach((f, i) => {
          const t = W.check.theta[i];
          success = Math.max(success, Math.abs(MODEL.exact(t, W.truth, W.queries) - W.check.success[i]));
          internal = Math.max(internal, Math.abs(MODEL.exact(t, W.learned, W.queries) - W.check.internal[i]));
          loss = Math.max(loss, Math.abs(MODEL.atomicLoss(t, W.learned) - W.check.loss[i]));
        });
        return { success, internal, loss };
      });
      console.log('max abs error vs saved checkpoints', JSON.stringify(fidelity));
      assert(fidelity.success < 1e-9 && fidelity.internal < 1e-9 && fidelity.loss < 1e-9);
      const replication = await page.evaluate(() => {
        const finals = D.worlds.map((W, k) => { const random = MODEL.mulberry(424242 + k); let t = MODEL.foldedStart(W.order); for (let i = 0; i < 3000; i++) t = MODEL.add(t, MODEL.practiceGrad(t, W.learned, W.queries, random, 'practice'), .1); return MODEL.exact(t, W.truth, W.queries); });
        const mean = finals.reduce((a, b) => a + b, 0) / finals.length, sd = Math.sqrt(finals.reduce((a, b) => a + (b - mean) ** 2, 0) / (finals.length - 1));
        return { mean, sem: sd / Math.sqrt(finals.length), above90: finals.filter(f => f > .9).length, saved: D.savedMean };
      });
      console.log('JS practice, 32 worlds x 3000 updates', JSON.stringify(replication));
      assert(Math.abs(replication.mean - replication.saved) < 3 * Math.SQRT2 * replication.sem);
      for (const id of ['task', 'data', 'generator', 'rules', 'step', 'eval', 'arrays']) await page.locator('#' + id).screenshot({ path: `${out}/${id}.png` });
      await page.locator('header').screenshot({ path: `${out}/header.png` });
      await page.locator('#run').click();
      await page.waitForFunction(() => document.getElementById('sandboxStatus').textContent.startsWith('update 3,000'), null, { timeout: 60000 });
      await page.locator('#sandbox').screenshot({ path: `${out}/sandbox.png` });
      await page.locator('#generator').screenshot({ path: `${out}/generator-after.png` });
      await page.locator('#contrastPair').click();
      await page.locator('#step').screenshot({ path: `${out}/step-after.png` });
      console.log('sandbox', await page.locator('#sandboxStatus').textContent());
    }
    await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
    await page.close();
  }
  await browser.close();
  console.log(out);
})();
