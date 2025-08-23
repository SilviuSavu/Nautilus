import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ 
    headless: false,
    devtools: true,
    slowMo: 1000
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  // Listen for console messages
  page.on('console', msg => {
    console.log(`🔍 BROWSER CONSOLE [${msg.type()}]:`, msg.text());
  });
  
  // Listen for page errors
  page.on('pageerror', error => {
    console.log('❌ PAGE ERROR:', error.message);
  });
  
  // Listen for network requests
  page.on('request', request => {
    console.log(`📡 REQUEST: ${request.method()} ${request.url()}`);
  });
  
  // Listen for failed network requests
  page.on('requestfailed', request => {
    console.log(`❌ FAILED REQUEST: ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });
  
  // Listen for responses
  page.on('response', response => {
    if (!response.ok()) {
      console.log(`❌ FAILED RESPONSE: ${response.status()} ${response.url()}`);
    }
  });
  
  console.log('🌐 Navigating to http://localhost:3000...');
  
  try {
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    console.log('✅ Page loaded successfully');
    
    // Take screenshot of the initial state
    await page.screenshot({ 
      path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/debug-initial-state.png',
      fullPage: true
    });
    console.log('📸 Screenshot saved: debug-initial-state.png');
    
    // Get page title
    const title = await page.title();
    console.log(`📄 Page title: "${title}"`);
    
    // Get the HTML content
    const html = await page.content();
    console.log(`📝 HTML length: ${html.length} characters`);
    
    // Check if there are any visible elements
    const bodyText = await page.textContent('body');
    console.log(`📝 Body text length: ${bodyText ? bodyText.length : 0} characters`);
    
    // Look for specific elements we expect
    const elements = {
      'Navigation': 'nav',
      'Dashboard': '[data-testid="dashboard"]',
      'Chart Container': '[data-testid="chart-container"]',
      'Error Message': '[role="alert"], .error, .error-message',
      'Loading Indicator': '.loading, .spinner, [data-testid="loading"]'
    };
    
    console.log('\n🔍 Checking for expected elements:');
    for (const [name, selector] of Object.entries(elements)) {
      try {
        const element = await page.locator(selector).first();
        const count = await element.count();
        console.log(`  ${count > 0 ? '✅' : '❌'} ${name}: ${count} found`);
        
        if (count > 0 && name === 'Error Message') {
          const errorText = await element.textContent();
          console.log(`    📝 Error text: "${errorText}"`);
        }
      } catch (e) {
        console.log(`  ❌ ${name}: Error checking - ${e.message}`);
      }
    }
    
    // Check for React error boundary
    const reactError = await page.locator('text="Something went wrong"').count();
    if (reactError > 0) {
      console.log('❌ React Error Boundary detected!');
    }
    
    // Wait a bit to see if anything loads
    console.log('\n⏳ Waiting 5 seconds to see if content loads...');
    await page.waitForTimeout(5000);
    
    // Take another screenshot
    await page.screenshot({ 
      path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/debug-after-wait.png',
      fullPage: true
    });
    console.log('📸 Screenshot after wait saved: debug-after-wait.png');
    
    // Check network tab for failed requests
    console.log('\n🌐 Checking for any backend connectivity...');
    
    try {
      const response = await page.request.get('http://localhost:3000');
      console.log(`✅ Frontend server responded: ${response.status()}`);
    } catch (e) {
      console.log(`❌ Frontend server check failed: ${e.message}`);
    }
    
    try {
      const response = await page.request.get('http://localhost:8001/health');
      console.log(`✅ Backend health check: ${response.status()}`);
      const healthData = await response.json();
      console.log('📊 Backend health data:', JSON.stringify(healthData, null, 2));
    } catch (e) {
      console.log(`❌ Backend health check failed: ${e.message}`);
    }
    
    // Check environment variables visible to the frontend
    const envVars = await page.evaluate(() => {
      return {
        VITE_API_BASE_URL: window.ENV?.VITE_API_BASE_URL || 'Not set',
        VITE_WS_URL: window.ENV?.VITE_WS_URL || 'Not set'
      };
    });
    console.log('\n🔧 Environment variables:', envVars);
    
    console.log('\n✅ Inspection complete. Check the screenshots and console output above.');
    console.log('Press any key to close the browser...');
    
    // Keep browser open for manual inspection
    await new Promise(resolve => {
      process.stdin.once('data', resolve);
    });
    
  } catch (error) {
    console.log('❌ Navigation failed:', error.message);
    
    // Try to take a screenshot anyway
    try {
      await page.screenshot({ 
        path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/debug-error-state.png',
        fullPage: true
      });
      console.log('📸 Error state screenshot saved: debug-error-state.png');
    } catch (screenshotError) {
      console.log('❌ Could not take error screenshot:', screenshotError.message);
    }
  }
  
  await browser.close();
})();