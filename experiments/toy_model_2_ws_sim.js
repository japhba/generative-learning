// Two-cause wake-sleep in the plane, with sleep, with recognition frozen, and the autoencoder baseline. Shared by the report page (inlined by
// toy_model_2_ws_report.py) and its node check. World: z uniform on {-1,+1}^2, x = z + sigma*eps (A = I).
// Matrices are 2x2 arrays [m00, m01, m10, m11] (row-major); the generator maps z to x = W z + c, the recognizer x to R x + b.
const WS = (() => {
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function normal(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const sign = r => r() < 0.5 ? -1 : 1;
  const rot = t => [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];

  function world(seed, sigma, n = 256) {
    const r = rng(seed), X = [], Z = [];
    for (let i = 0; i < n; i++) { const z = [sign(r), sign(r)]; Z.push(z); X.push([z[0] + sigma * normal(r), z[1] + sigma * normal(r)]); }
    return {X, Z, sigma};
  }

  // Start: generator rotated by theta0 from the world's axes; recognizer at its sleep fixed point for that generator.
  function learner(theta0, sigma) {
    const W = rot(theta0), k = 1 / sigma ** 2;
    return {W, c: [0, 0], R: [W[0] * k, W[2] * k, W[1] * k, W[3] * k], b: [0, 0]};
  }

  function read(L, x, r) {  // one sample from the recognizer q(z | x)
    const a0 = L.R[0] * x[0] + L.R[1] * x[1] + L.b[0], a1 = L.R[2] * x[0] + L.R[3] * x[1] + L.b[1];
    return [r() < 1 / (1 + Math.exp(-2 * a0)) ? 1 : -1, r() < 1 / (1 + Math.exp(-2 * a1)) ? 1 : -1];
  }

  function wake(L, w, r, lr, batch) {  // recognise stored foreign samples, delta rule on the generator
    const dW = [0, 0, 0, 0], dc = [0, 0];
    for (let n = 0; n < batch; n++) {
      const x = w.X[Math.floor(r() * w.X.length)], z = read(L, x, r);
      const e0 = x[0] - L.W[0] * z[0] - L.W[1] * z[1] - L.c[0], e1 = x[1] - L.W[2] * z[0] - L.W[3] * z[1] - L.c[1];
      dW[0] += e0 * z[0]; dW[1] += e0 * z[1]; dW[2] += e1 * z[0]; dW[3] += e1 * z[1]; dc[0] += e0; dc[1] += e1;
    }
    for (let i = 0; i < 4; i++) L.W[i] += lr * dW[i] / batch;
    L.c[0] += lr * dc[0] / batch; L.c[1] += lr * dc[1] / batch;
  }

  function sleep(L, sigma, r, lr, batch) {  // dream from the own prior, render, delta rule on the recognizer
    const dR = [0, 0, 0, 0], db = [0, 0];
    for (let n = 0; n < batch; n++) {
      const z = [sign(r), sign(r)];
      const x = [L.W[0] * z[0] + L.W[1] * z[1] + L.c[0] + sigma * normal(r), L.W[2] * z[0] + L.W[3] * z[1] + L.c[1] + sigma * normal(r)];
      const g0 = z[0] - Math.tanh(L.R[0] * x[0] + L.R[1] * x[1] + L.b[0]), g1 = z[1] - Math.tanh(L.R[2] * x[0] + L.R[3] * x[1] + L.b[1]);
      dR[0] += g0 * x[0]; dR[1] += g0 * x[1]; dR[2] += g1 * x[0]; dR[3] += g1 * x[1]; db[0] += g0; db[1] += g1;
    }
    for (let i = 0; i < 4; i++) L.R[i] += lr * dR[i] / batch;
    for (let i = 0; i < 2; i++) L.b[i] += lr * db[i] / batch;
  }

  // One practice update: a wake batch on stored data, then a sleep batch (or, with recognition frozen, a second wake batch).
  function step(L, w, r, {lr = 0.02, sleepScale = 10, batch = 32, frozen = false} = {}) {
    wake(L, w, r, lr, batch);
    if (frozen) wake(L, w, r, lr, batch); else sleep(L, w.sigma, r, lr * sleepScale, batch);
  }

  // The autoencoder baseline (practice_disentangle.ae_step 'bae'): mean-field code m = tanh(R x + b), loss = the expected
  // squared error under the factorial code, ||x - W m - c||^2 + sum_k (1 - m_k^2) ||w_k||^2; gradient steps on W, c (rate lr)
  // and, backpropagated through the code (scaled by 1 / sigma^2), on R, b (rate lr * rate). Never dreams.
  function aeBatch(L, w, r, lr, rate, batch) {
    const dW = [0, 0, 0, 0], dc = [0, 0], dR = [0, 0, 0, 0], db = [0, 0], v = [0, 0], s2 = w.sigma ** 2;
    const n2 = [L.W[0] ** 2 + L.W[2] ** 2, L.W[1] ** 2 + L.W[3] ** 2];  // squared lengths of the generator's columns
    for (let n = 0; n < batch; n++) {
      const x = w.X[Math.floor(r() * w.X.length)];
      const m = [Math.tanh(L.R[0] * x[0] + L.R[1] * x[1] + L.b[0]), Math.tanh(L.R[2] * x[0] + L.R[3] * x[1] + L.b[1])];
      const e0 = x[0] - L.W[0] * m[0] - L.W[1] * m[1] - L.c[0], e1 = x[1] - L.W[2] * m[0] - L.W[3] * m[1] - L.c[1];
      dW[0] += e0 * m[0]; dW[1] += e0 * m[1]; dW[2] += e1 * m[0]; dW[3] += e1 * m[1]; dc[0] += e0; dc[1] += e1;
      for (let k = 0; k < 2; k++) {
        const vk = 1 - m[k] ** 2, g = (L.W[k] * e0 + L.W[2 + k] * e1 + m[k] * n2[k]) / s2 * vk;
        v[k] += vk; dR[2 * k] += g * x[0]; dR[2 * k + 1] += g * x[1]; db[k] += g;
      }
    }
    for (let i = 0; i < 4; i++) L.W[i] += lr * (dW[i] - L.W[i] * v[i % 2]) / batch;
    for (let i = 0; i < 4; i++) L.R[i] += lr * rate * dR[i] / batch;
    for (let k = 0; k < 2; k++) { L.c[k] += lr * dc[k] / batch; L.b[k] += lr * rate * db[k] / batch; }
  }
  // One update of the autoencoder: two batches of supplied samples, the same number of batches as a practice update.
  function aeStep(L, w, r, {lr = 0.02, rate = 0.05, batch = 32} = {}) { aeBatch(L, w, r, lr, rate, batch); aeBatch(L, w, r, lr, rate, batch); }

  // Rotation angle (degrees) of the nearest rotation to the columns of M, folded to (-45, 45]: every multiple of 90 degrees,
  // and every sign flip, leaves the binary prior unchanged, so all of them count as aligned.
  function angle(M) {
    let [m00, m01, m10, m11] = M;
    if (m00 * m11 - m01 * m10 < 0) { m01 = -m01; m11 = -m11; }
    const t = Math.atan2(m10 - m01, m00 + m11) * 180 / Math.PI;
    return ((t + 45) % 90 + 90) % 90 - 45;
  }
  const genAngle = L => angle(L.W), recAngle = L => angle([L.R[0], L.R[2], L.R[1], L.R[3]]);  // recognizer: its rows as axes

  // Expected log-likelihood per world sample of the generator W = rotation(theta) (orthonormal, c = 0), relative to the
  // aligned generator: a mixture of four Gaussians, integrated over the noise on a deterministic grid.
  function landscape(sigma, thetas, grid = 41, span = 4.5) {
    const pts = [], h = 2 * span / (grid - 1); let tot = 0;
    for (let i = 0; i < grid; i++) for (let j = 0; j < grid; j++) { const u = -span + i * h, v = -span + j * h, wt = Math.exp(-(u * u + v * v) / 2); pts.push([u, v, wt]); tot += wt; }
    const Z = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
    const ll = t => {
      const W = rot(t); let s = 0;
      for (const zt of Z) for (const [u, v, wt] of pts) {
        const x0 = zt[0] + sigma * u, x1 = zt[1] + sigma * v; let m = -Infinity; const l = [];
        for (const z of Z) { const d0 = x0 - W[0] * z[0] - W[1] * z[1], d1 = x1 - W[2] * z[0] - W[3] * z[1], q = -(d0 * d0 + d1 * d1) / (2 * sigma * sigma); l.push(q); if (q > m) m = q; }
        s += wt * (m + Math.log(l.reduce((acc, q) => acc + Math.exp(q - m), 0) / 4));
      }
      return s / (4 * tot);
    };
    const ref = ll(0);
    return thetas.map(t => ll(t * Math.PI / 180) - ref);
  }

  return {rng, normal, world, learner, step, aeStep, genAngle, recAngle, landscape};
})();
if (typeof module !== 'undefined') module.exports = WS;
