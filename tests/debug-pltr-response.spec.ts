import { test, expect } from '@playwright/test'

test('debug PLTR API response in detail', async ({ page }) => {
  // Enable all console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  // Navigate to the dashboard
  await page.goto('http://localhost:3000')
  
  // Go to the Instrument Search tab
  await page.click('text=Instrument Search')
  await page.waitForTimeout(1000)
  
  // Click on the search input
  const searchInput = page.locator('input[placeholder*="Search instruments"]')
  await searchInput.click()
  
  // Type PLTR slowly to trigger search
  await searchInput.type('PLTR', { delay: 100 })
  
  // Wait a bit for the search to complete
  await page.waitForTimeout(3000)
  
  // Take a screenshot of what's displayed
  await page.screenshot({ path: 'pltr-search-results.png', fullPage: true })
  
  // Check if results are shown
  const resultsContainer = page.locator('.ant-list')
  const resultItems = await resultsContainer.locator('.ant-list-item').count()
  
  console.log(`Found ${resultItems} result items in the UI`)
  
  // Check what text is shown
  const allText = await page.textContent('body')
  console.log(`Page text contains EURUSD: ${allText?.includes('EURUSD')}`)
  console.log(`Page text contains PLTR: ${allText?.includes('PLTR')}`)
})