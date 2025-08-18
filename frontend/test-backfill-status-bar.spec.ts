import { test, expect } from '@playwright/test';

test('Historical Data Backfill Status Bar functionality', async ({ page }) => {
  // Enable console logging to see API calls
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('response', response => console.log('API RESPONSE:', response.url(), response.status()));

  // Navigate to the dashboard
  await page.goto('http://localhost:3000');
  
  // Wait for the page to load
  await page.waitForTimeout(2000);

  // Verify the Historical Data Backfill Status card is present
  const backfillCard = page.locator('text=Historical Data Backfill Status');
  await expect(backfillCard).toBeVisible();
  console.log('✅ Backfill status card is visible');

  // Check if the card contains expected elements
  const statisticsElements = [
    'Queue Size',
    'Completed',
    'Failed',
    'Overall Progress'
  ];

  for (const element of statisticsElements) {
    const statistic = page.locator(`text=${element}`).first();
    await expect(statistic).toBeVisible();
    console.log(`✅ Found statistic: ${element}`);
  }

  // Check for Active Requests statistic specifically
  const activeRequestsStat = page.locator('.ant-statistic-title:has-text("Active Requests")');
  await expect(activeRequestsStat).toBeVisible();
  console.log('✅ Found statistic: Active Requests');

  // Check for action buttons
  const startButton = page.locator('text=Start Priority Backfill');
  const stopButton = page.locator('text=Stop Backfill');
  const refreshButton = page.locator('text=Refresh Status');

  await expect(startButton).toBeVisible();
  await expect(stopButton).toBeVisible();
  await expect(refreshButton).toBeVisible();
  console.log('✅ All action buttons are visible');

  // Test refresh functionality
  await refreshButton.click();
  await page.waitForTimeout(1000);
  console.log('✅ Refresh button clicked');

  // Check if progress circle is present
  const progressCircle = page.locator('.ant-progress-circle').first();
  await expect(progressCircle).toBeVisible();
  console.log('✅ Progress circle is visible');

  // Check if there are any active backfill requests showing
  const activeRequestsTable = page.locator('text=Active Requests:');
  if (await activeRequestsTable.isVisible()) {
    console.log('✅ Active requests table is visible');
    
    // Check for table headers
    const tableHeaders = [
      'Symbol',
      'Timeframe', 
      'Status',
      'Success Count',
      'Errors',
      'Progress'
    ];

    for (const header of tableHeaders) {
      const headerElement = page.locator(`th:has-text("${header}")`);
      if (await headerElement.isVisible()) {
        console.log(`✅ Found table header: ${header}`);
      }
    }
  } else {
    console.log('ℹ️  No active requests table (no active backfill in progress)');
  }

  // Test the Start Priority Backfill button (if it's enabled)
  if (await startButton.isEnabled()) {
    console.log('ℹ️  Start Priority Backfill button is enabled');
    
    // Click to start backfill
    await startButton.click();
    await page.waitForTimeout(2000);
    console.log('✅ Start Priority Backfill button clicked');

    // Wait a bit for status to update
    await page.waitForTimeout(3000);

    // Check if status changed to running
    const runningBadge = page.locator('text=Running');
    if (await runningBadge.isVisible()) {
      console.log('✅ Backfill status changed to Running');
    }
  } else {
    console.log('ℹ️  Start Priority Backfill button is disabled (may already be running)');
  }

  // Check if the data is updating by waiting and checking again
  await page.waitForTimeout(5000);
  await refreshButton.click();
  await page.waitForTimeout(1000);
  console.log('✅ Status refreshed after waiting');

  // Take a screenshot for verification
  await page.screenshot({ path: 'test-backfill-status-bar.png', fullPage: true });
  console.log('✅ Screenshot saved as test-backfill-status-bar.png');

  // Verify no error states in the UI
  const errorElements = page.locator('.ant-alert-error, .ant-message-error, .error');
  const errorCount = await errorElements.count();
  expect(errorCount).toBe(0);
  console.log('✅ No error states found in UI');

  console.log('\n🎉 All backfill status bar tests passed!');
});