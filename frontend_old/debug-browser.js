// Simple browser test to check if React is loading
const puppeteer = require('puppeteer');

(async () => {
  console.log('🔍 Checking if frontend React app is loading...');
  
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  
  // Capture console logs
  page.on('console', msg => {
    console.log(`BROWSER: ${msg.type()} - ${msg.text()}`);
  });
  
  // Capture errors
  page.on('pageerror', error => {
    console.log(`PAGE ERROR: ${error.message}`);
  });
  
  // Navigate to frontend
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  
  // Wait a bit for React to load
  await page.waitForTimeout(5000);
  
  // Check if Dashboard loaded
  const title = await page.title();
  console.log(`📄 Page title: ${title}`);
  
  // Check if root has content
  const rootContent = await page.evaluate(() => {
    const root = document.getElementById('root');
    return {
      hasContent: root && root.innerHTML.length > 0,
      contentLength: root ? root.innerHTML.length : 0,
      preview: root ? root.innerHTML.substring(0, 200) : 'NO ROOT'
    };
  });
  
  console.log(`🎯 Root content: ${JSON.stringify(rootContent, null, 2)}`);
  
  // Look for Dashboard elements
  const dashboardElements = await page.evaluate(() => {
    return {
      dashboardTitle: !!document.querySelector('h2'),
      antTabs: !!document.querySelector('.ant-tabs'),
      performanceTab: !!document.querySelector('*[class*="ant-tabs-tab"]:contains("Performance")')
    };
  });
  
  console.log(`📊 Dashboard elements: ${JSON.stringify(dashboardElements, null, 2)}`);
  
  // Take a screenshot
  await page.screenshot({ path: 'frontend-debug.png', fullPage: true });
  console.log('📸 Screenshot saved as frontend-debug.png');
  
  if (rootContent.hasContent) {
    console.log('✅ Frontend React app is LOADING CONTENT');
  } else {
    console.log('❌ Frontend React app is BLANK - content not loading');
  }
  
  await browser.close();
})().catch(console.error);