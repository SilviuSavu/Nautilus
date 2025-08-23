import { test, expect } from '@playwright/test';

test('Fix frontend display issue with comprehensive debugging', async ({ page }) => {
  console.log('🚀 Starting comprehensive frontend debugging...');
  
  // Collect all console messages and errors
  const consoleMessages: string[] = [];
  const errors: string[] = [];
  const networkFailures: string[] = [];
  
  // Set up listeners
  page.on('console', msg => {
    const text = `[${msg.type()}] ${msg.text()}`;
    consoleMessages.push(text);
    console.log(`📋 Console: ${text}`);
  });

  page.on('pageerror', error => {
    const errorText = `PAGE ERROR: ${error.message}`;
    errors.push(errorText);
    console.log(`❌ ${errorText}`);
  });

  page.on('requestfailed', request => {
    const failed = `FAILED: ${request.url()} - ${request.failure()?.errorText}`;
    networkFailures.push(failed);
    console.log(`🌐 ${failed}`);
  });
  
  // Navigate to main page
  console.log('🔍 Step 1: Navigate to main page');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
  
  // Take initial screenshot
  await page.screenshot({ path: 'debug-step1-initial.png', fullPage: true });
  console.log('📸 Screenshot 1: Initial page load');
  
  // Wait a bit for any delayed scripts
  await page.waitForTimeout(3000);
  
  // Check what's actually in the page
  const bodyHTML = await page.innerHTML('body');
  console.log('📄 Body HTML length:', bodyHTML.length);
  
  const bodyText = await page.textContent('body');
  console.log('📄 Body text:', bodyText?.substring(0, 200) + '...');
  
  // Check if root element exists
  const rootExists = await page.locator('#root').count();
  console.log('🎯 Root element count:', rootExists);
  
  if (rootExists > 0) {
    const rootHTML = await page.innerHTML('#root');
    console.log('🎯 Root innerHTML length:', rootHTML.length);
    console.log('🎯 Root content preview:', rootHTML.substring(0, 200));
  }
  
  // Check if any React content is visible
  const reactContent = await page.locator('text=React').count();
  const nautilusContent = await page.locator('text=Nautilus').count();
  const loadingContent = await page.locator('text=Loading').count();
  const errorContent = await page.locator('text=ERROR').count();
  
  console.log('🔍 Content check:');
  console.log(`  - React: ${reactContent}`);
  console.log(`  - Nautilus: ${nautilusContent}`);
  console.log(`  - Loading: ${loadingContent}`);
  console.log(`  - Error: ${errorContent}`);
  
  // Take screenshot after waiting
  await page.screenshot({ path: 'debug-step2-after-wait.png', fullPage: true });
  console.log('📸 Screenshot 2: After waiting');
  
  // Try to interact with the page - force a reload
  console.log('🔄 Step 2: Force reload');
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  
  // Take screenshot after reload
  await page.screenshot({ path: 'debug-step3-after-reload.png', fullPage: true });
  console.log('📸 Screenshot 3: After reload');
  
  // Check if debug.html works
  console.log('🔍 Step 3: Test debug.html');
  await page.goto('http://localhost:3000/debug.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  
  const debugContent = await page.textContent('body');
  console.log('🧪 Debug page content:', debugContent?.substring(0, 200));
  
  await page.screenshot({ path: 'debug-step4-debug-page.png', fullPage: true });
  console.log('📸 Screenshot 4: Debug page');
  
  // Check if React is working on debug page
  const debugReactWorking = await page.locator('text=React is Working').count();
  console.log('🧪 Debug page React working:', debugReactWorking > 0);
  
  // Try to manually inject a simple React app
  console.log('🔧 Step 4: Try manual React injection');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
  
  // Inject simple test content
  await page.evaluate(() => {
    const root = document.getElementById('root');
    if (root) {
      root.innerHTML = '<div style="padding: 20px; background: green; color: white; font-size: 24px;">🎯 MANUAL INJECTION SUCCESSFUL</div>';
    } else {
      document.body.innerHTML = '<div style="padding: 20px; background: red; color: white; font-size: 24px;">❌ NO ROOT ELEMENT FOUND</div>';
    }
  });
  
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'debug-step5-manual-injection.png', fullPage: true });
  console.log('📸 Screenshot 5: Manual injection test');
  
  // Try to load React manually
  console.log('🔧 Step 5: Try manual React loading');
  await page.evaluate(async () => {
    try {
      // Clear everything first
      document.body.innerHTML = '<div id="root"></div><div id="status">Loading React manually...</div>';
      
      // Try to import React manually
      const React = await import('/node_modules/.vite/deps/react.js?v=d2c8bf89');
      const ReactDOM = await import('/node_modules/.vite/deps/react-dom_client.js?v=d2c8bf89');
      
      document.getElementById('status')!.innerHTML = 'React modules loaded, creating app...';
      
      const root = ReactDOM.createRoot(document.getElementById('root')!);
      root.render(React.createElement('div', {
        style: { padding: '20px', background: 'blue', color: 'white', fontSize: '24px' }
      }, '🚀 MANUAL REACT WORKING!'));
      
      document.getElementById('status')!.innerHTML = 'React manually rendered!';
    } catch (error) {
      document.body.innerHTML = `<div style="padding: 20px; background: red; color: white;">Manual React failed: ${error.message}</div>`;
    }
  });
  
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'debug-step6-manual-react.png', fullPage: true });
  console.log('📸 Screenshot 6: Manual React test');
  
  // Summary
  console.log('\n=== DEBUGGING SUMMARY ===');
  console.log(`Console messages: ${consoleMessages.length}`);
  console.log(`Page errors: ${errors.length}`);
  console.log(`Network failures: ${networkFailures.length}`);
  
  if (errors.length > 0) {
    console.log('\n❌ ERRORS:');
    errors.forEach(error => console.log(error));
  }
  
  if (networkFailures.length > 0) {
    console.log('\n🌐 FAILED REQUESTS:');
    networkFailures.forEach(req => console.log(req));
  }
  
  console.log('\n📋 RECENT CONSOLE MESSAGES:');
  consoleMessages.slice(-10).forEach(msg => console.log(msg));
});