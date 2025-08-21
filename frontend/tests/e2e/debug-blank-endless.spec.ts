import { test, expect } from '@playwright/test';

test.describe('ENDLESS LOOP DEBUG - Blank Frontend Fix', () => {
  test('debug blank page until completely fixed', async ({ page }) => {
    let attempt = 1;
    const maxAttempts = 50; // Safety limit
    
    while (attempt <= maxAttempts) {
      console.log(`🔍 DEBUG ATTEMPT ${attempt}/${maxAttempts}`);
      
      // Capture ALL console messages
      const consoleMessages: string[] = [];
      page.on('console', msg => {
        const message = `${msg.type().toUpperCase()}: ${msg.text()}`;
        consoleMessages.push(message);
        console.log(`BROWSER ${message}`);
      });
      
      // Capture ALL errors
      const pageErrors: string[] = [];
      page.on('pageerror', error => {
        pageErrors.push(error.message);
        console.log(`❌ PAGE ERROR: ${error.message}`);
        console.log(`❌ STACK: ${error.stack}`);
      });
      
      // Capture network failures
      page.on('requestfailed', request => {
        console.log(`❌ NETWORK FAILED: ${request.url()} - ${request.failure()?.errorText}`);
      });
      
      // Navigate and wait
      console.log(`⏳ Loading http://localhost:3000...`);
      await page.goto('http://localhost:3000', { 
        waitUntil: 'domcontentloaded',
        timeout: 10000 
      });
      
      // Wait for potential React loading
      await page.waitForTimeout(8000);
      
      // Check body content
      const bodyContent = await page.textContent('body');
      const bodyLength = bodyContent?.length || 0;
      console.log(`📏 Body content length: ${bodyLength}`);
      
      // Check if root element has content
      const rootElement = page.locator('#root');
      const rootContent = await rootElement.textContent();
      const rootLength = rootContent?.length || 0;
      console.log(`🎯 Root content length: ${rootLength}`);
      
      // Check for Dashboard elements
      const dashboardTitle = await page.locator('h2:has-text("NautilusTrader Dashboard")').count();
      const antTabs = await page.locator('.ant-tabs').count();
      const performanceTab = await page.locator('text=Performance Monitoring').count();
      
      console.log(`📊 Dashboard elements - Title: ${dashboardTitle}, Tabs: ${antTabs}, Performance: ${performanceTab}`);
      
      // Take screenshot for this attempt
      await page.screenshot({ 
        path: `debug-attempt-${attempt}.png`, 
        fullPage: true 
      });
      
      // Check if we have meaningful content
      if (bodyLength > 200 && dashboardTitle > 0 && antTabs > 0) {
        console.log(`✅ SUCCESS! Frontend is working after ${attempt} attempts`);
        console.log(`✅ Dashboard loaded with ${dashboardTitle} title, ${antTabs} tabs, ${performanceTab} performance tabs`);
        
        // Final verification - click Performance tab
        if (performanceTab > 0) {
          await page.click('text=Performance Monitoring');
          await page.waitForTimeout(3000);
          await page.screenshot({ path: 'success-performance-tab.png' });
          console.log(`✅ Performance tab clicked successfully`);
        }
        
        return; // Exit the loop - SUCCESS!
      }
      
      // Log what we found instead
      console.log(`❌ ATTEMPT ${attempt} FAILED:`);
      console.log(`   Body length: ${bodyLength} (need > 200)`);
      console.log(`   Dashboard title: ${dashboardTitle} (need > 0)`);
      console.log(`   Ant tabs: ${antTabs} (need > 0)`);
      console.log(`   Performance tab: ${performanceTab}`);
      
      if (pageErrors.length > 0) {
        console.log(`❌ JavaScript Errors Found:`);
        pageErrors.forEach((error, i) => console.log(`   ${i + 1}. ${error}`));
      }
      
      if (consoleMessages.length === 0) {
        console.log(`❌ NO CONSOLE MESSAGES - JavaScript may not be loading at all`);
      }
      
      // Try to fix common issues based on what we found
      if (attempt === 1 && pageErrors.length === 0 && consoleMessages.length === 0) {
        console.log(`🔧 ATTEMPT ${attempt + 1}: JavaScript not loading - check Vite server`);
      }
      
      attempt++;
      
      if (attempt <= maxAttempts) {
        console.log(`⏳ Waiting 2 seconds before next attempt...`);
        await page.waitForTimeout(2000);
      }
    }
    
    // If we get here, we failed all attempts
    console.log(`❌ FAILED after ${maxAttempts} attempts - Frontend is broken`);
    throw new Error(`Frontend still blank after ${maxAttempts} debugging attempts`);
  });
});