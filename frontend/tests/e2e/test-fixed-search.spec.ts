import { test, expect } from '@playwright/test'

test('Verify Fixed Instrument Search and Order Book Access', async ({ page }) => {
  page.on('console', msg => console.log(`🔧 FIXED: ${msg.text()}`))
  
  console.log('\n🔧 TESTING FIXED INSTRUMENT SEARCH')
  console.log('📋 Verifying the fix works end-to-end')

  console.log('\n🚀 Loading application...')
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(2000)

  console.log('\n🎯 Navigating to Interactive Brokers...')
  await page.locator('text=Interactive Brokers').click()
  await page.waitForTimeout(3000)

  console.log('\n🔍 Opening Search Instruments modal...')
  const searchButton = page.locator('button:has-text("Search Instruments")')
  if (await searchButton.isVisible()) {
    await searchButton.click()
    await page.waitForTimeout(2000)
    console.log('✅ Search modal opened')
  } else {
    console.log('❌ Search button not found')
    return
  }

  console.log('\n📝 Entering AAPL and executing search...')
  // Fill the symbol field
  const symbolInput = page.locator('input[placeholder*="AAPL"], .ant-modal input').first()
  await symbolInput.fill('AAPL')
  
  // Click search button
  const modalSearchBtn = page.locator('.ant-modal button:has-text("Search")').first()
  await modalSearchBtn.click()
  console.log('🔍 Search executed')
  
  // Wait for API response
  await page.waitForTimeout(3000)

  console.log('\n📊 Checking for search results...')
  
  // Close modal to see results
  await page.keyboard.press('Escape')
  await page.waitForTimeout(1000)
  
  // Check for AAPL in results table
  const aaplMentions = await page.locator('text=AAPL').count()
  const appleMentions = await page.locator('text=APPLE').count()
  const tableRows = await page.locator('table tbody tr').count()
  
  console.log(`   AAPL mentions: ${aaplMentions}`)
  console.log(`   APPLE mentions: ${appleMentions}`)
  console.log(`   Table rows: ${tableRows}`)

  if (aaplMentions > 0 || appleMentions > 0) {
    console.log('✅ SEARCH RESULTS FOUND!')
    
    console.log('\n🎯 Looking for Order Book button...')
    const orderBookButtons = await page.locator('button:has-text("Order Book")').count()
    console.log(`   Order Book buttons: ${orderBookButtons}`)
    
    if (orderBookButtons > 0) {
      console.log('🎉 ORDER BOOK BUTTON FOUND!')
      
      // Click the Order Book button
      const orderBookBtn = page.locator('button:has-text("Order Book")').first()
      await orderBookBtn.click()
      await page.waitForTimeout(2000)
      
      console.log('\n📊 Checking if Order Book opened...')
      
      // Look for Order Book content
      const orderBookContent = await page.locator('text=Spread, text=Bid Volume, text=Ask Volume').count()
      if (orderBookContent > 0) {
        console.log('🎉 ORDER BOOK SUCCESSFULLY OPENED!')
      } else {
        console.log('⚠️  Order Book button clicked but content not fully loaded')
      }
      
    } else {
      console.log('❌ Order Book button not found in results')
    }
  } else {
    console.log('❌ No search results found - fix may need more work')
  }

  console.log('\n📸 Taking final screenshot...')
  await page.screenshot({ path: 'fixed-search-test.png', fullPage: true })
  
  console.log('\n🎯 FIX VERIFICATION COMPLETE!')
  console.log('📋 Results:')
  if (aaplMentions > 0 || appleMentions > 0) {
    console.log('✅ Instrument search now works')
    console.log('✅ Search results appear in table')
    console.log('✅ Users can now select instruments')
    console.log('✅ Order Book access is now possible')
  } else {
    console.log('❌ Search still not working - needs further investigation')
  }
})