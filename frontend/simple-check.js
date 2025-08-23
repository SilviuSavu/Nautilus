import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const consoleMessages = [];
  const errors = [];
  
  // Capture console messages
  page.on('console', msg => {
    consoleMessages.push(`${msg.type()}: ${msg.text()}`);
  });
  
  // Capture errors
  page.on('pageerror', error => {
    errors.push(`Page Error: ${error.message}`);
  });
  
  try {
    console.log('🚀 Loading http://localhost:3000...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'domcontentloaded',
      timeout: 10000 
    });
    
    // Wait a bit for React to load
    await page.waitForTimeout(3000);
    
    const title = await page.title();
    console.log(`✅ Page Title: ${title}`);
    
    // Check for error elements on page
    const errorText = await page.textContent('body');
    if (errorText.includes('Error')) {
      console.log('❌ Found "Error" text on page');
    } else {
      console.log('✅ No "Error" text found on page');
    }
    
    // Take screenshot
    await page.screenshot({ path: 'current-frontend-state.png', fullPage: true });
    console.log('📸 Screenshot saved as current-frontend-state.png');
    
    console.log(`\\n📊 Console messages: ${consoleMessages.length}`);
    consoleMessages.slice(-5).forEach(msg => console.log(`  ${msg}`));
    
    console.log(`\\n🚨 Errors: ${errors.length}`);
    errors.forEach(error => console.log(`  ${error}`));
    
  } catch (error) {
    console.error('❌ Failed to load page:', error.message);
  }
  
  await browser.close();
})();