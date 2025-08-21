import { test, expect } from '@playwright/test'

test('Safari Instrument Selection Reality Check', async ({ page }) => {
  // Configure for Safari-like experience
  await page.setViewportSize({ width: 1440, height: 900 })
  
  page.on('console', msg => console.log(`🌐 SAFARI: ${msg.text()}`))
  
  console.log('\n🌐 TESTING REAL INSTRUMENT SELECTION IN SAFARI-LIKE BROWSER')
  console.log('📋 This will show exactly what a user experiences')
  console.log('=' * 60)

  console.log('\n🚀 STEP 1: Loading the application as a real user would...')
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(3000)
  
  const pageTitle = await page.title()
  console.log(`📱 Page loaded: ${pageTitle}`)

  console.log('\n🎯 STEP 2: Navigating to Interactive Brokers tab...')
  const ibTab = page.locator('text=Interactive Brokers').first()
  if (await ibTab.isVisible()) {
    await ibTab.click()
    await page.waitForTimeout(3000)
    console.log('✅ IB tab clicked, dashboard loading...')
  } else {
    console.log('❌ IB tab not found!')
    return
  }

  console.log('\n🔍 STEP 3: Real user trying to find how to search for instruments...')
  
  // Show what a real user sees
  const visibleText = await page.locator('body').textContent()
  console.log(`📄 Page contains ${visibleText?.length} characters of content`)
  
  // Look for any search-related UI
  console.log('\n🔎 Looking for search functionality that a user would see:')
  
  const searchOptions = [
    { name: 'Search Input Field', selector: 'input[type="text"]' },
    { name: 'Search Button', selector: 'button:has-text("Search")' },
    { name: 'Instrument Search', selector: 'button:has-text("Search Instruments")' },
    { name: 'Symbol Input', selector: 'input[placeholder*="symbol" i]' },
    { name: 'AAPL Placeholder', selector: 'input[placeholder*="AAPL" i]' },
    { name: 'Search Icon', selector: '*[aria-label*="search" i]' }
  ]
  
  let foundSearchOptions = 0
  for (const option of searchOptions) {
    const count = await page.locator(option.selector).count()
    const visible = await page.locator(option.selector).first().isVisible().catch(() => false)
    
    if (count > 0 && visible) {
      console.log(`✅ ${option.name}: Found and visible (${count})`)
      foundSearchOptions++
      
      // Try to interact with it
      try {
        const element = page.locator(option.selector).first()
        await element.scrollIntoViewIfNeeded()
        await element.hover()
        console.log(`   👆 Hovering over ${option.name}`)
        await page.waitForTimeout(1000)
      } catch (e) {
        console.log(`   ⚠️  Could not interact with ${option.name}`)
      }
    } else {
      console.log(`❌ ${option.name}: Not found or not visible`)
    }
  }
  
  console.log(`\n📊 Total search options visible to user: ${foundSearchOptions}`)

  console.log('\n📝 STEP 4: Attempting to search for AAPL as a real user...')
  
  if (foundSearchOptions === 0) {
    console.log('❌ NO SEARCH FUNCTIONALITY VISIBLE TO USER!')
    console.log('🚨 This explains why users cannot select instruments')
    
    // Show what IS visible to the user
    console.log('\n👀 What the user actually sees - visible buttons:')
    const buttons = await page.locator('button:visible').allTextContents()
    buttons.forEach((button, index) => {
      if (button.trim()) {
        console.log(`   ${index + 1}. "${button.trim()}"`)
      }
    })
    
    console.log('\n👀 What the user actually sees - visible text inputs:')
    const inputs = await page.locator('input:visible').count()
    console.log(`   Found ${inputs} visible input fields`)
    
    if (inputs > 0) {
      for (let i = 0; i < inputs; i++) {
        const input = page.locator('input:visible').nth(i)
        const placeholder = await input.getAttribute('placeholder')
        const type = await input.getAttribute('type')
        console.log(`   ${i + 1}. Type: ${type}, Placeholder: "${placeholder || 'none'}"`)
      }
    }
  } else {
    console.log('✅ Search functionality found, testing it...')
    
    // Try to actually use the search
    const firstSearchOption = searchOptions.find(option => 
      page.locator(option.selector).first().isVisible()
    )
    
    if (firstSearchOption) {
      const element = page.locator(firstSearchOption.selector).first()
      
      if (firstSearchOption.name.includes('Input')) {
        console.log(`📝 Typing AAPL in ${firstSearchOption.name}...`)
        await element.click()
        await element.fill('AAPL')
        await page.waitForTimeout(2000)
      } else {
        console.log(`🖱️  Clicking ${firstSearchOption.name}...`)
        await element.click()
        await page.waitForTimeout(2000)
      }
    }
  }

  console.log('\n📊 STEP 5: Checking for instrument selection results...')
  
  // Look for any results or instrument data
  const resultIndicators = [
    'text=AAPL',
    'text=Apple',
    'table',
    '.instrument',
    '.result',
    'button:has-text("Order Book")'
  ]
  
  let resultsFound = 0
  for (const indicator of resultIndicators) {
    const count = await page.locator(indicator).count()
    if (count > 0) {
      console.log(`✅ Found ${indicator}: ${count} elements`)
      resultsFound++
    }
  }
  
  if (resultsFound === 0) {
    console.log('❌ NO INSTRUMENT RESULTS VISIBLE TO USER')
    console.log('🚨 User cannot see any way to select AAPL or any other instrument')
  }

  console.log('\n🔍 STEP 6: Looking for Order Book access...')
  
  const orderBookAccess = await page.locator('text=Order Book').count()
  console.log(`📋 "Order Book" text found: ${orderBookAccess} times`)
  
  if (orderBookAccess > 0) {
    const element = page.locator('text=Order Book').first()
    const isClickable = await element.evaluate(el => {
      const style = window.getComputedStyle(el)
      return style.cursor === 'pointer' || el.tagName === 'BUTTON' || el.tagName === 'A'
    })
    console.log(`🖱️  Order Book text is clickable: ${isClickable}`)
    
    if (isClickable) {
      console.log('🎯 Attempting to click Order Book...')
      await element.click()
      await page.waitForTimeout(2000)
      
      // Check what happens after clicking
      const newContent = await page.locator('body').textContent()
      const hasNewContent = newContent?.includes('Bid') || newContent?.includes('Ask') || newContent?.includes('Spread')
      console.log(`📊 Order Book content appeared: ${hasNewContent}`)
    }
  }

  console.log('\n🎯 STEP 7: Final user experience assessment...')
  
  await page.screenshot({ path: 'safari-user-experience.png', fullPage: true })
  
  console.log('\n🚨 REAL USER EXPERIENCE ANALYSIS:')
  console.log('=' * 50)
  
  if (foundSearchOptions === 0) {
    console.log('❌ CRITICAL ISSUE: User cannot search for instruments')
    console.log('   - No visible search input fields')
    console.log('   - No "Search Instruments" button visible')
    console.log('   - No way to enter AAPL or any symbol')
    console.log('')
    console.log('🔧 REQUIRED FIXES:')
    console.log('   1. Make instrument search UI visible and accessible')
    console.log('   2. Add clear search input field for symbols')
    console.log('   3. Add prominent "Search" or "Find Instrument" button')
    console.log('   4. Show search results in a clear table/list')
    console.log('   5. Add "Order Book" buttons to each search result')
  } else {
    console.log('✅ Search functionality is visible to users')
  }
  
  if (resultsFound === 0) {
    console.log('❌ CRITICAL ISSUE: No instrument results shown')
    console.log('   - User searches but sees no results')
    console.log('   - No instrument selection possible')
    console.log('   - Cannot access Order Book without instrument selection')
  }
  
  console.log('\n💡 USER JOURNEY SHOULD BE:')
  console.log('1. Go to IB Dashboard')
  console.log('2. See prominent search field')
  console.log('3. Type "AAPL" and press search')
  console.log('4. See AAPL in results table')
  console.log('5. Click "Order Book" button next to AAPL')
  console.log('6. See real-time order book visualization')
  
  console.log('\n📸 Screenshot saved: safari-user-experience.png')
  console.log('🎯 This shows exactly what real users see!')
})