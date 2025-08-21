import { test, expect } from '@playwright/test'

test('Debug Search Issue - Real Browser Console', async ({ page }) => {
  // Capture all console messages and network requests
  const consoleMessages: string[] = []
  const networkRequests: string[] = []
  const errors: string[] = []

  page.on('console', msg => {
    const message = `${msg.type()}: ${msg.text()}`
    consoleMessages.push(message)
    console.log(`🖥️  CONSOLE: ${message}`)
  })

  page.on('request', request => {
    const url = request.url()
    if (url.includes('search') || url.includes('instruments')) {
      networkRequests.push(`REQUEST: ${request.method()} ${url}`)
      console.log(`🌐 REQUEST: ${request.method()} ${url}`)
    }
  })

  page.on('response', async response => {
    const url = response.url()
    if (url.includes('search') || url.includes('instruments')) {
      const status = response.status()
      let body = 'Could not read body'
      try {
        if (response.headers()['content-type']?.includes('application/json')) {
          body = JSON.stringify(await response.json()).substring(0, 200) + '...'
        }
      } catch (e) {
        body = 'Body read error'
      }
      networkRequests.push(`RESPONSE: ${status} ${url} - ${body}`)
      console.log(`🔍 RESPONSE: ${status} ${url}`)
      console.log(`📊 BODY: ${body}`)
    }
  })

  page.on('pageerror', error => {
    errors.push(error.message)
    console.log(`❌ PAGE ERROR: ${error.message}`)
  })

  console.log('\n🔍 DEBUGGING SEARCH ISSUE')
  console.log('📋 Will capture all console, network, and error data')

  await page.goto('http://localhost:3000')
  await page.waitForTimeout(2000)

  console.log('\n🎯 Navigating to IB Dashboard...')
  await page.locator('text=Interactive Brokers').click()
  await page.waitForTimeout(3000)

  console.log('\n🔍 Opening search modal...')
  const searchButton = page.locator('button:has-text("Search Instruments")').first()
  
  if (await searchButton.isVisible()) {
    await searchButton.click()
    await page.waitForTimeout(1000)
    console.log('✅ Search modal opened')
    
    console.log('\n📝 Entering PLTR and searching...')
    const symbolInput = page.locator('.ant-modal input[placeholder*="AAPL"], .ant-modal input').first()
    await symbolInput.fill('PLTR')
    
    console.log('🔍 Clicking search button...')
    const modalSearchBtn = page.locator('.ant-modal button:has-text("Search")').first()
    await modalSearchBtn.click()
    
    console.log('⏳ Waiting 5 seconds for API response...')
    await page.waitForTimeout(5000)
    
  } else {
    console.log('❌ Search button not found')
    
    // Show what buttons ARE visible
    const allButtons = await page.locator('button:visible').allTextContents()
    console.log('📊 Visible buttons:')
    allButtons.forEach((btn, i) => {
      if (btn.trim()) console.log(`   ${i+1}. "${btn.trim()}"`)
    })
  }

  console.log('\n📊 DEBUGGING SUMMARY:')
  console.log('=' * 50)
  
  console.log(`\n🖥️  Console Messages (${consoleMessages.length}):`)
  consoleMessages.slice(-10).forEach(msg => console.log(`   ${msg}`))
  
  console.log(`\n🌐 Network Requests (${networkRequests.length}):`)
  networkRequests.forEach(req => console.log(`   ${req}`))
  
  console.log(`\n❌ Errors (${errors.length}):`)
  errors.forEach(err => console.log(`   ${err}`))
  
  if (networkRequests.length === 0) {
    console.log('\n🚨 CRITICAL: NO NETWORK REQUESTS MADE!')
    console.log('   This means the search button click is not triggering the API call')
    console.log('   The frontend code is not executing the search function')
  }
  
  console.log('\n🔧 Next steps based on findings:')
  if (errors.length > 0) {
    console.log('   Fix JavaScript errors first')
  }
  if (networkRequests.length === 0) {
    console.log('   Debug why handleInstrumentSearch is not being called')
  }
  
  await page.screenshot({ path: 'debug-search-state.png', fullPage: true })
})