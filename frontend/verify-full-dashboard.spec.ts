import { test, expect } from '@playwright/test'

test('Verify full Nautilus Trading Platform dashboard is working on port 3001', async ({ page }) => {
  console.log('🚀 Navigating to full dashboard on port 3001...')
  
  await page.goto('http://localhost:3001', { waitUntil: 'networkidle' })
  
  // Wait for the dashboard to load
  await page.waitForTimeout(3000)
  
  // Take screenshot of the full dashboard
  await page.screenshot({ 
    path: 'full-dashboard-verification.png', 
    fullPage: true 
  })
  console.log('📸 Screenshot saved: full-dashboard-verification.png')
  
  // Check for dashboard title
  const title = await page.textContent('h2')
  console.log('📋 Dashboard title:', title)
  
  // Check for tabs
  const tabs = await page.locator('[role="tab"]').count()
  console.log('📑 Number of tabs found:', tabs)
  
  // Check for key elements
  const hasSystemTab = await page.locator('text=System Overview').count()
  const hasNautilusTab = await page.locator('text=NautilusTrader Engine').count()
  const hasInstrumentTab = await page.locator('text=Instrument Search').count()
  
  console.log('✅ Dashboard verification:')
  console.log(`  - System Overview tab: ${hasSystemTab > 0 ? 'Found' : 'Missing'}`)
  console.log(`  - NautilusTrader Engine tab: ${hasNautilusTab > 0 ? 'Found' : 'Missing'}`)
  console.log(`  - Instrument Search tab: ${hasInstrumentTab > 0 ? 'Found' : 'Missing'}`)
  
  // Verify this is NOT the simple test app
  const hasSimpleTest = await page.locator('text=React App Working!').count()
  console.log(`  - Simple test app detected: ${hasSimpleTest > 0 ? 'YES (ERROR)' : 'NO (CORRECT)'}`)
  
  expect(hasSystemTab).toBeGreaterThan(0)
  expect(hasNautilusTab).toBeGreaterThan(0)
  expect(hasInstrumentTab).toBeGreaterThan(0)
  expect(hasSimpleTest).toBe(0) // Should NOT have the simple test app
})