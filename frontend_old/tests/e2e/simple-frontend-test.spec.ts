/**
 * Simple Frontend Test
 */

import { test, expect } from '@playwright/test';

test('Simple frontend functionality test', async ({ page }) => {
  console.log('🧪 Starting simple frontend test...');
  
  await page.goto('http://localhost:3000');
  
  // Wait for content without specific selectors
  await page.waitForTimeout(5000);
  
  // Get page content
  const content = await page.textContent('body');
  console.log('📄 Page loaded with content length:', content?.length || 0);
  
  // Take screenshot
  await page.screenshot({ path: 'simple-frontend-working.png' });
  
  // Verify we have content
  expect(content!.length).toBeGreaterThan(1000);
  
  console.log('✅ Frontend test passed!');
});