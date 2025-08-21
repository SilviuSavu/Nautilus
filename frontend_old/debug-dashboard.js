import { chromium } from 'playwright';

(async () => {
  console.log('🔍 Debugging dashboard crash...');
  
  const browser = await chromium.launch({ 
    headless: false,  // Show the browser
    slowMo: 2000      // Slow down actions
  });
  const page = await browser.newPage();
  
  // Listen for ALL console logs and errors
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    console.log(`🌐 [${type.toUpperCase()}]:`, text);
  });
  
  page.on('pageerror', err => {
    console.log('💥 PAGE ERROR:', err.message);
    console.log('📍 Stack:', err.stack);
  });
  
  try {
    console.log('🌐 Navigating to dashboard...');
    await page.goto('http://localhost:3000/dashboard');
    
    console.log('⏱️  Waiting for page to load...');
    await page.waitForTimeout(10000);
    
    // Check what we can see
    console.log('📊 Checking dashboard state...');
    
    const title = await page.title();
    console.log('📄 Page title:', title);
    
    const url = page.url();
    console.log('🔗 Current URL:', url);
    
    // Try to get the page content
    const bodyText = await page.locator('body').textContent();
    console.log('📝 Body text preview:', bodyText?.substring(0, 200) + '...');
    
    // Take a screenshot
    await page.screenshot({ path: 'debug-dashboard.png', fullPage: true });
    console.log('📸 Screenshot saved as debug-dashboard.png');
    
  } catch (error) {
    console.log('❌ Error occurred:', error.message);
    await page.screenshot({ path: 'debug-error.png', fullPage: true });
  }
  
  console.log('⏸️  Keeping browser open for manual inspection...');
  console.log('🔍 Press Ctrl+C to close when done');
  
  // Keep the browser open for manual inspection
  await page.waitForTimeout(300000); // 5 minutes
  
  await browser.close();
})();