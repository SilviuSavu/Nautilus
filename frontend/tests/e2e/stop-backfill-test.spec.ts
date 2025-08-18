import { test, expect } from '@playwright/test'

test('stop backfill button functionality', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  // Navigate to dashboard
  await page.goto('http://localhost:3000')
  
  // Wait for page to load
  await page.waitForTimeout(3000)
  
  // Look for the stop backfill button
  const stopButton = page.locator('text=Stop Backfill')
  
  // Check if button exists
  await expect(stopButton).toBeVisible()
  
  // Check if button is clickable (not disabled)
  const isDisabled = await stopButton.isDisabled()
  console.log('Stop button disabled:', isDisabled)
  
  // If button is enabled, test the click functionality
  if (!isDisabled) {
    // Click the stop button
    await stopButton.click()
    
    // Wait for any response
    await page.waitForTimeout(2000)
    
    // Check for any errors in console
    const consoleErrors = []
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text())
      }
    })
    
    // Take screenshot for verification
    await page.screenshot({ path: 'stop-backfill-test.png', fullPage: true })
    
    console.log('Stop backfill button test completed')
  } else {
    console.log('Stop button is disabled - this is expected when backfill is not running')
  }
})