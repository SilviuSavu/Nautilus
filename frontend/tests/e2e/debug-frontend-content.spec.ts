/**
 * Debug Frontend Content Test - Find out why frontend is blank
 */

import { test, expect } from '@playwright/test';

test.describe('Frontend Content Debug', () => {
  test('Debug what is actually rendering in the frontend', async ({ page }) => {
    console.log('🔍 Starting frontend debug analysis...');
    
    // Capture all console logs and errors
    page.on('console', msg => {
      console.log(`BROWSER CONSOLE [${msg.type()}]:`, msg.text());
    });
    
    page.on('pageerror', error => {
      console.log('PAGE ERROR:', error.message);
    });
    
    page.on('response', response => {
      if (response.status() >= 400) {
        console.log(`❌ HTTP ERROR: ${response.url()} - ${response.status()}`);
      }
    });
    
    console.log('📍 Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000');
    
    console.log('⏱️ Waiting for page to load...');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    
    // Get detailed content information
    const html = await page.content();
    console.log('📄 Full HTML length:', html.length);
    
    const bodyText = await page.textContent('body');
    console.log('📝 Body text content:', bodyText);
    console.log('📏 Body text length:', bodyText?.length || 0);
    
    // Check for React mounting
    const reactElements = await page.locator('[data-reactroot], #root').count();
    console.log('⚛️ React root elements found:', reactElements);
    
    // Check for any visible text
    const visibleText = await page.locator('body *').allTextContents();
    console.log('👁️ All visible text elements:', visibleText);
    
    // Check for specific elements
    const divCount = await page.locator('div').count();
    console.log('📦 Total div elements:', divCount);
    
    // Check for navigation or dashboard elements
    const navElements = await page.locator('nav, .ant-menu, .ant-layout').count();
    console.log('🧭 Navigation elements found:', navElements);
    
    // Take a screenshot for visual inspection
    await page.screenshot({ 
      path: 'debug-frontend-blank.png',
      fullPage: true 
    });
    console.log('📸 Screenshot saved: debug-frontend-blank.png');
    
    // Check for any JavaScript errors in the network
    const failedRequests = [];
    page.on('response', response => {
      if (response.status() >= 400) {
        failedRequests.push(`${response.url()} - ${response.status()}`);
      }
    });
    
    console.log('🌐 Failed requests:', failedRequests);
    
    // Check if main app component is rendered
    const appComponent = await page.locator('#root > *').count();
    console.log('🏗️ App components inside root:', appComponent);
    
    // Log the actual HTML structure
    const bodyHTML = await page.locator('body').innerHTML();
    console.log('🔍 Body HTML structure:');
    console.log(bodyHTML.substring(0, 500) + '...');
  });
});