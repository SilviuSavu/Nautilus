/**
 * Debug Performance Tab Loading Issue
 */
import { test, expect } from '@playwright/test';

test('Debug Performance Tab Loading', async ({ page }) => {
  console.log('🔍 Debugging Performance Tab Loading Issue');
  
  // Capture all console messages and errors
  const consoleMessages: string[] = [];
  const errorMessages: string[] = [];
  
  page.on('console', msg => {
    const text = msg.text();
    consoleMessages.push(text);
    console.log('BROWSER:', text);
  });
  
  page.on('pageerror', error => {
    const errorText = error.message;
    errorMessages.push(errorText);
    console.log('PAGE ERROR:', errorText);
  });
  
  // Navigate to the page
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);
  
  // Take screenshot of initial state
  await page.screenshot({ path: 'debug-initial-dashboard.png' });
  
  console.log('📋 Initial console messages:', consoleMessages.length);
  console.log('❌ Initial error messages:', errorMessages.length);
  
  // Click Performance Monitoring tab
  console.log('🎯 Clicking Performance tab...');
  await page.click('text=Perform');
  await page.waitForTimeout(5000); // Wait longer to see if it loads
  
  // Take screenshot after clicking
  await page.screenshot({ path: 'debug-after-performance-click.png' });
  
  console.log('📋 Console messages after click:', consoleMessages.slice(consoleMessages.length - 10));
  console.log('❌ Error messages after click:', errorMessages);
  
  // Check if PerformanceDashboard component is present
  const perfDashboardExists = await page.locator('.performance-dashboard').count();
  console.log('📊 Performance Dashboard elements found:', perfDashboardExists);
  
  // Check if ErrorBoundary is showing
  const errorBoundary = await page.locator('text*=Performance Monitoring Error').count();
  console.log('⚠️ Error boundary messages:', errorBoundary);
  
  // Check what's actually visible in the tab content
  const tabContent = await page.locator('[role="tabpanel"][aria-hidden="false"]').textContent();
  console.log('📄 Visible tab content preview:', tabContent?.substring(0, 200));
  
  // Look for any loading indicators
  const loadingSpinners = await page.locator('.ant-spin').count();
  console.log('⏳ Loading spinners:', loadingSpinners);
  
  // Check for any API calls being made
  const apiCalls: string[] = [];
  page.on('response', response => {
    if (response.url().includes('/api/')) {
      apiCalls.push(`${response.status()} ${response.url()}`);
      console.log('📡 API Call:', response.status(), response.url());
    }
  });
  
  // Wait a bit more to see if any delayed API calls happen
  await page.waitForTimeout(3000);
  
  console.log('📡 API calls made:', apiCalls);
  
  // Final screenshot
  await page.screenshot({ path: 'debug-performance-final.png' });
  
  // Report findings
  console.log('\n=== DEBUGGING SUMMARY ===');
  console.log('Console messages:', consoleMessages.length);
  console.log('Error messages:', errorMessages.length);
  console.log('Performance Dashboard found:', perfDashboardExists > 0);
  console.log('Error boundary triggered:', errorBoundary > 0);
  console.log('API calls made:', apiCalls.length);
  
  if (errorMessages.length > 0) {
    console.log('\n🚨 ERRORS DETECTED:');
    errorMessages.forEach((error, index) => {
      console.log(`${index + 1}. ${error}`);
    });
  }
});