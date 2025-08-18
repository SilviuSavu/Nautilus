import { test, expect } from '@playwright/test'

test('verify IB Gateway backfill prevents instrument search', async ({ page }) => {
  // Log all console messages and network requests
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  page.on('request', request => {
    if (request.url().includes('instruments') || request.url().includes('backfill')) {
      console.log(`🔵 REQUEST: ${request.method()} ${request.url()}`)
    }
  })
  
  page.on('response', async response => {
    if (response.url().includes('instruments') || response.url().includes('backfill')) {
      console.log(`🟢 RESPONSE: ${response.status()} ${response.url()}`)
    }
  })
  
  await page.goto('http://localhost:3000')
  await page.waitForLoadState('networkidle')
  
  // Navigate to system overview to check backfill status
  await page.click('text=System Overview')
  await page.waitForTimeout(2000)
  
  // Check if backfill is running by looking for the status indicator
  const backfillRunning = await page.locator('text=Running').count() > 0
  console.log(`🔍 Backfill currently running: ${backfillRunning}`)
  
  // Test 1: Try instrument search when backfill might be running
  await page.click('text=Instrument Search')
  await page.waitForTimeout(1000)
  
  const searchInput = page.locator('input[placeholder*="Search instruments"]')
  await searchInput.fill('PLTR')
  await page.waitForTimeout(5000) // Give more time for potential timeout
  
  const searchResults1 = page.locator('[class*="ant-list-item"]')
  const count1 = await searchResults1.count()
  console.log(`🔍 Search results when backfill status unknown: ${count1}`)
  
  // Test 2: Stop backfill if running and try search again
  if (backfillRunning) {
    console.log(`🛑 Stopping backfill to test instrument search...`)
    await page.click('text=System Overview')
    await page.waitForTimeout(1000)
    
    // Find and click stop backfill button
    const stopButton = page.locator('button', { hasText: 'Stop Backfill' })
    if (await stopButton.count() > 0) {
      await stopButton.click()
      console.log(`🛑 Clicked stop backfill button`)
      await page.waitForTimeout(3000) // Wait for backfill to stop
    }
    
    // Now try instrument search again
    await page.click('text=Instrument Search')
    await page.waitForTimeout(1000)
    
    await searchInput.fill('')
    await searchInput.fill('PLTR')
    await page.waitForTimeout(5000)
    
    const searchResults2 = page.locator('[class*="ant-list-item"]')
    const count2 = await searchResults2.count()
    console.log(`🔍 Search results after stopping backfill: ${count2}`)
    
    // The hypothesis is that stopping backfill should allow search to work
    if (count2 > count1) {
      console.log(`✅ HYPOTHESIS CONFIRMED: Stopping backfill improved search results (${count1} → ${count2})`)
    } else {
      console.log(`❌ HYPOTHESIS NOT CONFIRMED: No improvement in search results`)
    }
  } else {
    console.log(`ℹ️ Backfill not running, cannot test conflict scenario`)
  }
  
  // Test 3: Start backfill and verify search becomes unavailable
  console.log(`🚀 Starting backfill to test conflict...`)
  await page.click('text=System Overview')
  await page.waitForTimeout(1000)
  
  const startButton = page.locator('button', { hasText: 'Start IB Gateway Backfill' })
  if (await startButton.count() > 0) {
    await startButton.click()
    console.log(`🚀 Started IB Gateway backfill`)
    await page.waitForTimeout(2000) // Wait for backfill to start
    
    // Try instrument search while backfill is running
    await page.click('text=Instrument Search')
    await page.waitForTimeout(1000)
    
    await searchInput.fill('')
    await searchInput.fill('PLTR')
    await page.waitForTimeout(5000)
    
    const searchResults3 = page.locator('[class*="ant-list-item"]')
    const count3 = await searchResults3.count()
    console.log(`🔍 Search results while backfill is running: ${count3}`)
    
    if (count3 === 0) {
      console.log(`✅ HYPOTHESIS CONFIRMED: Running backfill prevents instrument search`)
    } else {
      console.log(`❌ HYPOTHESIS NOT CONFIRMED: Search still works during backfill`)
    }
  }
  
  // Take screenshot for evidence
  await page.screenshot({ path: 'ib-gateway-conflict-test.png' })
  
  // Always pass the test - we're collecting data
  expect(true).toBe(true)
})