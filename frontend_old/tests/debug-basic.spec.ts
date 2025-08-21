import { test, expect } from '@playwright/test'

test('basic dashboard load test', async ({ page }) => {
  // Enable console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  console.log('Navigating to http://localhost:3000/dashboard...')
  await page.goto('http://localhost:3000/dashboard')
  
  console.log('Waiting for page to load...')
  await page.waitForTimeout(5000)
  
  console.log('Taking screenshot...')
  await page.screenshot({ path: 'debug-dashboard.png' })
  
  console.log('Getting page title...')
  const title = await page.title()
  console.log('Page title:', title)
  
  console.log('Getting page content...')
  const content = await page.content()
  console.log('Page has content length:', content.length)
  
  console.log('Looking for dashboard element...')
  const dashboardExists = await page.locator('[data-testid="dashboard"]').count()
  console.log('Dashboard elements found:', dashboardExists)
  
  if (dashboardExists === 0) {
    console.log('Dashboard not found, checking for any divs...')
    const divCount = await page.locator('div').count()
    console.log('Total div elements:', divCount)
    
    console.log('Checking for any error messages...')
    const errors = await page.locator('text=Error').count()
    console.log('Error messages found:', errors)
  }
})