import { test, expect } from '@playwright/test'

test('debug PLTR search step by step', async ({ page }) => {
  // Intercept and log ALL network requests and responses
  page.on('request', request => {
    if (request.url().includes('instruments')) {
      console.log(`🔵 REQUEST: ${request.method()} ${request.url()}`)
      console.log(`🔵 REQUEST HEADERS:`, request.headers())
    }
  })
  
  page.on('response', async response => {
    if (response.url().includes('instruments')) {
      console.log(`🟢 RESPONSE: ${response.status()} ${response.url()}`)
      console.log(`🟢 RESPONSE HEADERS:`, response.headers())
      try {
        const responseText = await response.text()
        console.log(`🟢 RESPONSE BODY:`, responseText.substring(0, 500))
      } catch (e) {
        console.log('Could not read response body')
      }
    }
  })
  
  // Log request failures
  page.on('requestfailed', request => {
    if (request.url().includes('instruments')) {
      console.log(`❌ REQUEST FAILED: ${request.method()} ${request.url()}`)
      console.log(`❌ FAILURE REASON:`, request.failure()?.errorText)
    }
  })

  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  await page.goto('http://localhost:3000')
  await page.waitForLoadState('networkidle')
  
  // Navigate to instrument search tab
  await page.click('text=Instrument Search')
  await page.waitForTimeout(1000)
  
  // Type PLTR in search
  const searchInput = page.locator('input[placeholder*="Search instruments"]')
  await searchInput.fill('PLTR')
  
  // Wait for search to complete
  await page.waitForTimeout(3000)
  
  // Check for any visible results
  const searchResults = page.locator('[class*="ant-list-item"]')
  const count = await searchResults.count()
  
  console.log(`FRONTEND: Found ${count} search results for PLTR`)
  
  // Check if any result contains PLTR
  if (count > 0) {
    for (let i = 0; i < Math.min(count, 3); i++) {
      const resultText = await searchResults.nth(i).textContent()
      console.log(`FRONTEND RESULT ${i + 1}: ${resultText}`)
    }
  }
  
  // Take a screenshot for debugging
  await page.screenshot({ path: 'pltr-search-debug.png' })
  
  // The test should pass regardless so we can see the debug output
  expect(true).toBe(true)
})