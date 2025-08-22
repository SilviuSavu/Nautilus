import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000');
  
  console.log('Waiting for page to load...');
  await page.waitForTimeout(5000);
  
  console.log('Taking screenshot...');
  await page.screenshot({ path: 'frontend-debug.png', fullPage: true });
  
  console.log('Getting page content...');
  const content = await page.content();
  console.log('Page HTML length:', content.length);
  
  const bodyText = await page.locator('body').textContent();
  console.log('Body text:', bodyText);
  console.log('Body text length:', bodyText?.length || 0);
  
  const errors = await page.evaluate(() => {
    return window.console.error.toString();
  });
  
  console.log('Console errors:', errors);
  
  await browser.close();
})();