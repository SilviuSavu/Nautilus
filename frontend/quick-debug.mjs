import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: false });
const page = await browser.newPage();

page.on('console', msg => console.log('CONSOLE:', msg.text()));
page.on('pageerror', error => console.log('ERROR:', error.message));

await page.goto('http://localhost:3000');
await page.waitForTimeout(5000);

const rootHTML = await page.locator('#root').innerHTML();
console.log('ROOT CONTENT:', rootHTML);

const hasReact = await page.evaluate(() => !!window.React);
console.log('React loaded:', hasReact);

await page.waitForTimeout(10000);
await browser.close();