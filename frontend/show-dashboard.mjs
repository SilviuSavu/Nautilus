import { chromium } from 'playwright';

async function showDashboard() {
  console.log('🌐 Opening browser to show you the dashboard...');
  
  const browser = await chromium.launch({ 
    headless: false,  // Keep browser open so you can see it
    devtools: true,   // Open dev tools
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--start-maximized'
    ]
  });
  
  const page = await browser.newPage();
  
  // Track any errors
  page.on('console', msg => {
    console.log(`🖥️  BROWSER: ${msg.text()}`);
  });
  
  page.on('pageerror', error => {
    console.log(`❌ PAGE ERROR: ${error.message}`);
  });
  
  try {
    console.log('📍 Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for React to render...');
    await page.waitForTimeout(3000);
    
    // Check if dashboard is visible
    const dashboard = await page.locator('[data-testid="dashboard"]');
    const isVisible = await dashboard.isVisible();
    
    console.log(`👀 Dashboard visible: ${isVisible}`);
    
    if (isVisible) {
      const text = await dashboard.textContent();
      console.log(`📄 Dashboard content: ${text?.substring(0, 200)}...`);
    } else {
      console.log('🔍 Checking what is visible...');
      const bodyText = await page.locator('body').textContent();
      console.log(`📄 Body content: ${bodyText?.substring(0, 200)}...`);
    }
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'dashboard-screenshot.png', 
      fullPage: true 
    });
    console.log('📸 Screenshot saved as dashboard-screenshot.png');
    
    // Highlight the dashboard if it exists
    try {
      await page.locator('[data-testid="dashboard"]').highlight();
    } catch (e) {
      console.log('⚠️  Could not highlight dashboard element');
    }
    
    console.log('🖱️  Browser is open - you should now see the dashboard!');
    console.log('🔧 Check the browser window that just opened');
    console.log('⌛ Browser will stay open for 60 seconds for you to inspect...');
    
    // Keep browser open for inspection
    await page.waitForTimeout(60000);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    // Still take screenshot on error
    try {
      await page.screenshot({ path: 'error-screenshot.png' });
      console.log('📸 Error screenshot saved as error-screenshot.png');
    } catch (e) {
      console.log('Could not take error screenshot');
    }
  }
  
  await browser.close();
  console.log('🏁 Browser session ended');
}

showDashboard().catch(console.error);