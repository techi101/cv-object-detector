// End-to-end browser test: opens the web app in a real (headless) Microsoft
// Edge, uploads an image with on-device detection, runs live webcam detection
// with Chromium's fake camera, checks the "Server" option, and checks there is
// no sideways scrolling at phone width. Prints what it measured.
//
// Setup (once):   npm install playwright-core
// Run:            uvicorn app:app --port 8010      (in another terminal)
//                 node tests/e2e_browser.js http://127.0.0.1:8010/ path/to/bus.jpg
// Needs Microsoft Edge installed (it uses the browser already on the machine).
// Also works against the live site: node tests/e2e_browser.js https://cv-object-detector-zrgy.onrender.com/ bus.jpg
const { chromium } = require('playwright-core');
const path = require('path');
const URL = process.argv[2] || 'http://127.0.0.1:8010/';
const IMG = path.resolve(process.argv[3] || 'bus.jpg');
const SHOTS = path.resolve('results');
require('fs').mkdirSync(SHOTS, { recursive: true });

(async () => {
    const browser = await chromium.launch({
        channel: 'msedge', headless: true,
        args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    });
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    await ctx.grantPermissions(['camera']);
    const page = await ctx.newPage();
    const logs = [];
    page.on('console', m => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));

    const t0 = Date.now();
    await page.goto(URL);
    await page.waitForFunction(() => /Model ready|unavailable/.test(
        document.getElementById('engine-status').textContent), null, { timeout: 180000 });
    console.log('status:', await page.textContent('#engine-status'), `(${Date.now() - t0} ms incl. page load)`);
    console.log('crossOriginIsolated:', await page.evaluate(() => self.crossOriginIsolated));
    await page.screenshot({ path: path.join(SHOTS, 'home.png') });

    // 1. Upload on device
    await page.setInputFiles('#file-input', IMG);
    await page.waitForSelector('#results:not(.hidden)', { timeout: 60000 });
    console.log('upload: count', await page.textContent('#total-count'),
        '| timing', await page.textContent('#timing'),
        '| classes', (await page.$$eval('.class-item', els => els.map(e => e.textContent.replace(/\s+/g, ' ').trim()))).join(', '));
    await page.waitForTimeout(1000); await page.screenshot({ path: path.join(SHOTS, 'upload.png'), fullPage: true });

    // Repeat a few times for a warm number
    const times = [];
    for (let i = 0; i < 5; i++) {
        await page.click('text=Analyze Another');
        await page.setInputFiles('#file-input', IMG);
        await page.waitForSelector('#results:not(.hidden)');
        times.push(await page.textContent('#timing'));
    }
    console.log('warm uploads:', times.join(' | '));

    // 2. Live webcam on device
    await page.click('text=Analyze Another');
    await page.click('#tab-webcam');
    await page.click('text=Enable Camera');
    await page.waitForSelector('#live-btn', { state: 'visible' });
    await page.click('#live-btn');
    await page.waitForTimeout(6000);
    console.log('live: fps', await page.textContent('#live-fps'), '| objects', await page.textContent('#live-count'));
    await page.screenshot({ path: path.join(SHOTS, 'live.png') });
    await page.click('#live-btn');

    // 3. Server path still works
    await page.click('#tab-upload');
    await page.click('#engine-server');
    await page.setInputFiles('#file-input', IMG);
    await page.waitForSelector('#results:not(.hidden)', { timeout: 120000 });
    console.log('server: count', await page.textContent('#total-count'), '| timing', await page.textContent('#timing'));

    // 4. Phone width
    await page.click('text=Analyze Another');
    await page.click('#engine-device');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(SHOTS, 'mobile.png'), fullPage: true });
    console.log('mobile scrollWidth', await page.evaluate(() => document.documentElement.scrollWidth));

    console.log('--- console ---\n' + logs.filter(l => !/\[(debug|verbose)\]/.test(l)).join('\n'));
    await browser.close();
})().catch(e => { console.error('E2E FAILED:', e); process.exit(1); });
