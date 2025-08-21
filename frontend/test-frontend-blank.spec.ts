import { test, expect } from '@playwright/test';

test.describe('Frontend Blank Page Investigation', () => {
  test('investigate blank frontend page', async ({ page }) => {
    // Capture all console messages
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(`${msg.type()}: ${msg.text()}`);
      console.log(`BROWSER ${msg.type()}: ${msg.text()}`);
    });
    
    // Capture network errors
    page.on('requestfailed', request => {
      console.log(`NETWORK FAILED: ${request.url()} - ${request.failure()?.errorText}`);
    });
    
    console.log('🔍 Investigating blank frontend...');
    
    // Navigate to frontend
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    // Wait longer for React to load
    await page.waitForTimeout(5000);
    
    // Take screenshot first
    await page.screenshot({ path: 'frontend-blank-investigation.png', fullPage: true });
    
    // Check if page has any content at all
    const bodyText = await page.locator('body').textContent();
    console.log(`📄 Body text length: ${bodyText?.length || 0}`);
    console.log(`📄 Body content preview: ${bodyText?.substring(0, 200) || 'EMPTY'}`);
    
    // Check if React root element exists
    const rootElement = page.locator('#root');
    const rootExists = await rootElement.count();
    console.log(`🎯 Root element count: ${rootExists}`);
    
    if (rootExists > 0) {
      const rootContent = await rootElement.textContent();
      console.log(`🎯 Root content length: ${rootContent?.length || 0}`);
      console.log(`🎯 Root content: ${rootContent?.substring(0, 200) || 'EMPTY'}`);
    }
    
    // Check for specific React/Dashboard elements
    const dashboardTitle = page.locator('h2:has-text("NautilusTrader Dashboard")');
    const titleExists = await dashboardTitle.count();
    console.log(`📊 Dashboard title found: ${titleExists > 0}`);
    
    // Check for tabs
    const tabs = page.locator('.ant-tabs-tab');
    const tabCount = await tabs.count();
    console.log(`📑 Tab count: ${tabCount}`);
    
    // Check for any loading states
    const loading = page.locator('text=Loading');
    const loadingCount = await loading.count();
    console.log(`⏳ Loading indicators: ${loadingCount}`);
    
    // Check for error messages
    const errors = page.locator('.ant-alert-error, .error, [class*="error"]');
    const errorCount = await errors.count();
    console.log(`❌ Error elements: ${errorCount}`);
    
    if (errorCount > 0) {
      for (let i = 0; i < errorCount; i++) {
        const errorText = await errors.nth(i).textContent();
        console.log(`❌ Error ${i + 1}: ${errorText}`);
      }
    }
    
    // Print all console messages
    console.log(`📝 Total console messages: ${consoleMessages.length}`);
    consoleMessages.forEach((msg, i) => {
      console.log(`📝 ${i + 1}: ${msg}`);
    });
    
    // Final assessment
    if (bodyText && bodyText.length > 100) {
      console.log('✅ Frontend has content - NOT BLANK');
    } else {
      console.log('🚨 FRONTEND IS BLANK - NEEDS INVESTIGATION');
      
      // Additional debugging for blank page
      const htmlContent = await page.content();
      console.log(`📋 Full HTML length: ${htmlContent.length}`);
      console.log(`📋 HTML head: ${htmlContent.substring(0, 500)}`);
    }
  });
});