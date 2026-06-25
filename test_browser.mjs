import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
const errs = [];
p.on('console', m => errs.push(`[${m.type()}] ${m.text()}`));
p.on('pageerror', e => errs.push(`[pageerror] ${e.message}\n${e.stack||''}`));
await p.goto('file:///Users/bharat/repos/glm-art/output/fluvioglyph.html', { waitUntil: 'networkidle', timeout: 60000 });
await p.waitForTimeout(3000);
await p.screenshot({ path: 'output/shot_0_initial.png' });
// press play and advance
await p.click('#play');
await p.waitForTimeout(3000);
await p.screenshot({ path: 'output/shot_1_mid.png' });
// scrub near end
await p.fill('#scrub', '980');
await p.waitForTimeout(2000);
await p.screenshot({ path: 'output/shot_2_end.png' });
console.log('=== console ===');
console.log(errs.join('\n') || '(none)');
await b.close();
