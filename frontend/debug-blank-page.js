import { chromium } from 'playwright';

async function debugBlankPage() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  // Enable console logging
  page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  page.on('requestfailed', request => console.log('REQUEST FAILED:', request.url(), request.failure()?.errorText));
  
  try {
    console.log('🔍 Navigating to localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    console.log('📄 Current URL:', page.url());
    console.log('📝 Page title:', await page.title());
    
    // Check if there's any content
    const bodyText = await page.locator('body').textContent();
    console.log('📊 Body content length:', bodyText?.length || 0);
    
    // Check for specific elements
    const rootDiv = page.locator('#root');
    const rootExists = await rootDiv.count();
    console.log('🌳 Root div exists:', rootExists > 0);
    
    if (rootExists > 0) {
      const rootContent = await rootDiv.textContent();
      console.log('🌳 Root content:', rootContent?.substring(0, 200) || 'EMPTY');
    }
    
    // Check for React
    const reactDetected = await page.evaluate(() => {
      return !!(window.React || window.__REACT_DEVTOOLS_GLOBAL_HOOK__);
    });
    console.log('⚛️ React detected:', reactDetected);
    
    // Check for errors in console
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    // Wait a bit and check for dynamic content
    await page.waitForTimeout(3000);
    
    // Take a screenshot
    await page.screenshot({ path: 'debug-screenshot.png', fullPage: true });
    console.log('📸 Screenshot saved as debug-screenshot.png');
    
    // Check network requests
    const failedRequests = [];
    page.on('requestfailed', request => {
      failedRequests.push(`${request.url()}: ${request.failure()?.errorText}`);
    });
    
    // Get HTML content
    const htmlContent = await page.content();
    console.log('🌐 HTML structure:');
    console.log(htmlContent.substring(0, 1000));
    
    // Check if Vite dev server is running
    const viteCheck = await page.evaluate(() => {
      return !!window.__vite_plugin_react_preamble_installed__;
    });
    console.log('🔧 Vite detected:', viteCheck);
    
    // Look for specific error messages
    const errorElements = await page.locator('text=/error|Error|ERROR/').count();
    console.log('❌ Error elements found:', errorElements);
    
    if (errorElements > 0) {
      const errorTexts = await page.locator('text=/error|Error|ERROR/').allTextContents();
      console.log('❌ Error messages:', errorTexts);
    }
    
    console.log('\n🔧 DIAGNOSIS COMPLETE');
    console.log('Failed requests:', failedRequests);
    
    // Keep browser open for manual inspection
    console.log('\n🖱️ Browser will stay open for manual inspection...');
    await page.waitForTimeout(30000); // Wait 30 seconds
    
  } catch (error) {
    console.error('❌ Error during debugging:', error.message);
  } finally {
    await browser.close();
  }
}

debugBlankPage().catch(console.error);