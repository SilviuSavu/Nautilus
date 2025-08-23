import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ 
    headless: false,
    devtools: false,
    slowMo: 500
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  // Listen for console messages
  page.on('console', msg => {
    console.log(`🔍 CONSOLE [${msg.type()}]:`, msg.text());
  });
  
  // Listen for page errors
  page.on('pageerror', error => {
    console.log('❌ PAGE ERROR:', error.message);
  });
  
  // Listen for failed network requests
  page.on('requestfailed', request => {
    console.log(`❌ FAILED REQUEST: ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });
  
  console.log('🌐 Testing frontend at http://localhost:3000...');
  
  try {
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    console.log('✅ Page loaded successfully');
    
    // Take screenshot of the current state
    await page.screenshot({ 
      path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/frontend-after-fixes.png',
      fullPage: true
    });
    console.log('📸 Screenshot saved: frontend-after-fixes.png');
    
    // Check page title
    const title = await page.title();
    console.log(`📄 Page title: "${title}"`);
    
    // Check if there's actual content now
    const bodyText = await page.textContent('body');
    console.log(`📝 Body text length: ${bodyText ? bodyText.length : 0} characters`);
    if (bodyText && bodyText.length > 100) {
      console.log('📝 Body text preview:', bodyText.substring(0, 200) + '...');
    }
    
    // Look for key dashboard elements
    const elements = {
      'Dashboard Content': '[class*="dashboard"], [class*="Dashboard"]',
      'Ant Design Components': '.ant-card, .ant-button, .ant-tabs',
      'Charts': 'canvas, svg',
      'Navigation': 'nav, [role="navigation"]',
      'Error Messages': '[role="alert"], .error, .error-message'
    };
    
    console.log('\n🔍 Checking for dashboard elements:');
    for (const [name, selector] of Object.entries(elements)) {
      try {
        const count = await page.locator(selector).count();
        console.log(`  ${count > 0 ? '✅' : '❌'} ${name}: ${count} found`);
        
        if (count > 0 && name === 'Error Messages') {
          const errorTexts = await page.locator(selector).allTextContents();
          errorTexts.forEach(error => {
            console.log(`    📝 Error: "${error}"`);
          });
        }
      } catch (e) {
        console.log(`  ❌ ${name}: Error checking - ${e.message}`);
      }
    }
    
    // Check if environment variables are now accessible
    const envCheck = await page.evaluate(() => {
      const viteApiUrl = import.meta.env.VITE_API_BASE_URL;
      const viteWsUrl = import.meta.env.VITE_WS_URL;
      return {
        VITE_API_BASE_URL: viteApiUrl || 'Not available in import.meta.env',
        VITE_WS_URL: viteWsUrl || 'Not available in import.meta.env'
      };
    });
    console.log('\n🔧 Environment variables check:', envCheck);
    
    // Test backend connectivity through the frontend
    console.log('\n🌐 Testing backend connectivity through frontend...');
    try {
      const healthResponse = await page.evaluate(async () => {
        const response = await fetch('/health');
        if (response.ok) {
          return await response.json();
        }
        throw new Error(`HTTP ${response.status}`);
      });
      console.log('✅ Backend health via frontend proxy:', JSON.stringify(healthResponse));
    } catch (error) {
      console.log('❌ Backend health via frontend proxy failed:', error.message);
    }
    
    // Wait a bit more to see if content loads
    console.log('\n⏳ Waiting 5 seconds for dynamic content...');
    await page.waitForTimeout(5000);
    
    // Take final screenshot
    await page.screenshot({ 
      path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/frontend-final-state.png',
      fullPage: true
    });
    console.log('📸 Final screenshot saved: frontend-final-state.png');
    
    // Final content check
    const finalBodyText = await page.textContent('body');
    console.log(`📝 Final body text length: ${finalBodyText ? finalBodyText.length : 0} characters`);
    
    if (finalBodyText && finalBodyText.length > 1000) {
      console.log('🎉 SUCCESS: Frontend appears to be working with substantial content!');
    } else if (finalBodyText && finalBodyText.length > 100) {
      console.log('⚠️  PARTIAL: Frontend has some content but may not be fully loading');
    } else {
      console.log('❌ FAILURE: Frontend still appears mostly blank');
    }
    
    console.log('\n✅ Test complete. Check screenshots for visual confirmation.');
    
  } catch (error) {
    console.log('❌ Navigation failed:', error.message);
    
    await page.screenshot({ 
      path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/frontend-error-final.png',
      fullPage: true
    });
    console.log('📸 Error screenshot saved: frontend-error-final.png');
  }
  
  await browser.close();
  console.log('🏁 Browser closed. Test complete.');
})();