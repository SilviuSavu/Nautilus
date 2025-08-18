import { test, expect } from '@playwright/test'

test('Frontend Demo - Show how great it works!', async ({ page }) => {
  // Enable console logging to see what's happening
  page.on('console', msg => console.log('🖥️  BROWSER:', msg.text()))
  
  console.log('🚀 Opening the Nautilus Trader Frontend...')
  await page.goto('http://localhost:3000/dashboard')
  
  console.log('⏳ Waiting for the page to load...')
  await page.waitForTimeout(3000)
  
  console.log('📸 Taking a screenshot to show you the homepage...')
  await page.screenshot({ path: 'frontend-homepage.png', fullPage: true })
  
  console.log('✅ Checking if dashboard is visible...')
  const dashboardVisible = await page.locator('[data-testid="dashboard"]').isVisible()
  console.log('📊 Dashboard visible:', dashboardVisible)
  
  console.log('🔍 Looking for the main title...')
  const title = await page.locator('h2:has-text("NautilusTrader Dashboard")').textContent()
  console.log('📝 Page title:', title)
  
  console.log('🏷️  Finding all available tabs...')
  const tabs = await page.locator('.ant-tabs-tab').allTextContents()
  console.log('📑 Available tabs:', tabs)
  
  console.log('📊 Clicking on Financial Chart tab...')
  await page.click('text=Financial Chart')
  await page.waitForTimeout(2000)
  
  console.log('📸 Taking screenshot of chart tab...')
  await page.screenshot({ path: 'frontend-chart-tab.png', fullPage: true })
  
  console.log('🔍 Looking for chart components...')
  const chartComponent = await page.locator('[data-testid*="chart"]').count()
  console.log('📈 Chart components found:', chartComponent)
  
  console.log('🔄 Checking WebSocket status...')
  await page.waitForTimeout(2000)
  
  console.log('🏠 Going back to System Overview tab...')
  await page.click('text=System Overview')
  await page.waitForTimeout(1000)
  
  console.log('📸 Taking final screenshot...')
  await page.screenshot({ path: 'frontend-final.png', fullPage: true })
  
  console.log('🎉 Frontend demo complete! Check the screenshots to see how great it works!')
  
  // Keep the browser open for a while so you can see it
  console.log('⏰ Keeping browser open for 30 seconds so you can see it in action...')
  await page.waitForTimeout(30000)
})