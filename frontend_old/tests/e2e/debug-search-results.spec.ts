import { test, expect } from '@playwright/test'

test('debug what search actually returns', async ({ page }) => {
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  await page.goto('http://localhost:3000')
  await page.waitForLoadState('networkidle')
  
  await page.click('text=Instrument Search')
  await page.waitForTimeout(1000)
  
  // Search for PLTR
  const searchInput = page.locator('input[placeholder*="Search instruments"]')
  await searchInput.fill('PLTR')
  await page.waitForTimeout(2000)
  
  // Get actual text content of search results
  const searchResults = page.locator('[class*="ant-list-item"]')
  const count = await searchResults.count()
  
  console.log(`Total results for PLTR: ${count}`)
  
  // Log first 5 results to see what we actually get
  for (let i = 0; i < Math.min(count, 5); i++) {
    const resultText = await searchResults.nth(i).textContent()
    console.log(`Result ${i + 1}: ${resultText}`)
  }
  
  // Check if any result actually contains "PLTR"
  const pltrResults = page.locator('[class*="ant-list-item"]:has-text("PLTR")')
  const pltrCount = await pltrResults.count()
  console.log(`Results containing "PLTR": ${pltrCount}`)
  
  if (pltrCount === 0) {
    console.log('❌ NO PLTR FOUND - Search is broken!')
  } else {
    console.log('✅ PLTR found in results')
  }
})