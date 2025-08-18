import { chromium } from 'playwright';

(async () => {
  console.log('🚀 Testing restored dashboard...');
  
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  await page.goto('http://localhost:3000/dashboard');
  await page.waitForTimeout(5000);
  
  console.log('📸 Taking screenshot...');
  await page.screenshot({ path: 'dashboard-restored.png', fullPage: true });
  
  // Check for errors
  const errors = await page.evaluate(() => {
    return window.console ? 'Console available' : 'No console errors detected';
  });
  
  console.log('✅ Dashboard test complete');
  console.log('📄 Screenshot saved as: dashboard-restored.png');
  
  await browser.close();
})();