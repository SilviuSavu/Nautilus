const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0', timeout: 10000 });
  
  // Wait a bit for React to mount
  await page.waitForTimeout(3000);
  
  // Check if dashboard element exists
  const dashboardExists = await page.$('[data-testid="dashboard"]') !== null;
  console.log('Dashboard element exists:', dashboardExists);
  
  // Check if root has content
  const rootContent = await page.evaluate(() => document.getElementById('root').innerHTML);
  console.log('Root content length:', rootContent.length);
  console.log('Root has content:', rootContent.length > 0);
  
  if (rootContent.length < 100) {
    console.log('Root content:', rootContent);
  }
  
  await browser.close();
})();