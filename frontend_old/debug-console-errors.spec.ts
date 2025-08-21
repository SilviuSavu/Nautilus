import { test, expect } from '@playwright/test'

test('Debug console errors and check what is actually loading', async ({ page }) => {
  console.log('🔍 Debugging console errors and page content...')
  
  // Collect console messages and errors
  const consoleMessages: string[] = []
  const errors: string[] = []
  
  page.on('console', msg => {
    const text = `[${msg.type()}] ${msg.text()}`
    consoleMessages.push(text)
    console.log(`📋 Console: ${text}`)
  })

  page.on('pageerror', error => {
    const errorText = `PAGE ERROR: ${error.message}`
    errors.push(errorText)
    console.log(`❌ ${errorText}`)
  })

  page.on('requestfailed', request => {
    console.log(`🌐 FAILED REQUEST: ${request.url()} - ${request.failure()?.errorText}`)
  })
  
  // Navigate to the page
  await page.goto('http://localhost:3001', { waitUntil: 'domcontentloaded' })
  
  // Wait and see what happens
  await page.waitForTimeout(5000)
  
  // Check what's in the DOM
  const bodyHTML = await page.innerHTML('body')
  const bodyText = await page.textContent('body')
  
  console.log('📄 Body HTML:', bodyHTML.substring(0, 500))
  console.log('📄 Body text:', bodyText)
  
  // Check if React is running at all
  const hasReactRoot = await page.locator('#root').count()
  console.log('🎯 React root element count:', hasReactRoot)
  
  if (hasReactRoot > 0) {
    const rootContent = await page.innerHTML('#root')
    console.log('🎯 Root content:', rootContent)
  }
  
  // Take screenshot
  await page.screenshot({ path: 'debug-console-errors.png', fullPage: true })
  
  // Summary
  console.log('\n=== SUMMARY ===')
  console.log(`Total console messages: ${consoleMessages.length}`)
  console.log(`Total errors: ${errors.length}`)
  
  if (errors.length > 0) {
    console.log('\n❌ ERRORS FOUND:')
    errors.forEach(error => console.log(error))
  }
  
  console.log('\n📋 LAST 10 CONSOLE MESSAGES:')
  consoleMessages.slice(-10).forEach(msg => console.log(msg))
})