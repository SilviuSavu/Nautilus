import { test, expect } from '@playwright/test'

test('Demonstrate Correct User Flow for Search Access', async ({ page }) => {
  page.on('console', msg => console.log(`🖥️  BROWSER: ${msg.text()}`))
  
  console.log('\n🔧 FIXING THE USER FLOW ISSUE')
  console.log('📋 User needs to be on correct tab to see search')

  console.log('\n🚀 Loading application...')
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(2000)

  console.log('\n🎯 Going to Interactive Brokers...')
  await page.locator('text=Interactive Brokers').click()
  await page.waitForTimeout(3000)

  console.log('\n📊 Currently on Order Book tab - NO SEARCH BUTTON HERE')
  let searchButton = page.locator('button:has-text("Search Instruments")')
  let isVisible = await searchButton.isVisible()
  console.log(`   Search button visible on Order Book tab: ${isVisible}`)
  
  await page.screenshot({ path: 'order-book-tab-no-search.png', fullPage: true })

  console.log('\n🔍 Switching to Instrument Discovery tab...')
  await page.locator('text=Instrument Discovery').click()
  await page.waitForTimeout(1000)

  console.log('\n✅ NOW search button should be visible!')
  isVisible = await searchButton.isVisible()
  console.log(`   Search button visible on Instrument Discovery tab: ${isVisible}`)
  
  if (isVisible) {
    console.log('\n🎉 FOUND THE SEARCH BUTTON!')
    await page.screenshot({ path: 'instrument-discovery-with-search.png', fullPage: true })
    
    console.log('\n🔍 Clicking Search Instruments...')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    console.log('\n📝 Entering PLTR in search modal...')
    const symbolInput = page.locator('.ant-modal input[placeholder*="AAPL"], .ant-modal input').first()
    await symbolInput.fill('PLTR')
    
    console.log('\n🔍 Clicking search button in modal...')
    const modalSearchBtn = page.locator('.ant-modal button:has-text("Search")').first()
    await modalSearchBtn.click()
    
    console.log('\n⏳ Waiting for API response...')
    await page.waitForTimeout(3000)
    
    console.log('\n❌ Closing modal to see results...')
    await page.keyboard.press('Escape')
    await page.waitForTimeout(1000)
    
    console.log('\n📊 Checking for search results...')
    const pltrMentions = await page.locator('text=PLTR').count()
    const tableRows = await page.locator('table tbody tr').count()
    
    console.log(`   PLTR mentions: ${pltrMentions}`)
    console.log(`   Table rows: ${tableRows}`)
    
    if (pltrMentions > 0) {
      console.log('\n🎉 SEARCH WORKS! Looking for Order Book button...')
      const orderBookBtns = await page.locator('button:has-text("Order Book")').count()
      console.log(`   Order Book buttons found: ${orderBookBtns}`)
      
      if (orderBookBtns > 0) {
        console.log('\n🎉 ORDER BOOK BUTTON FOUND! User can now access order book!')
        
        // Click the Order Book button
        console.log('\n🖱️  Clicking Order Book button...')
        await page.locator('button:has-text("Order Book")').first().click()
        await page.waitForTimeout(2000)
        
        console.log('\n🎯 Should now switch to Order Book tab with selected instrument')
        const orderBookTab = page.locator('text=Order Book')
        await orderBookTab.click()
        await page.waitForTimeout(2000)
        
        console.log('\n📸 Taking final screenshot of working order book...')
        await page.screenshot({ path: 'working-order-book-final.png', fullPage: true })
        
        console.log('\n✅ SUCCESS! Complete user flow working!')
      } else {
        console.log('\n❌ Order Book button still not appearing in search results')
      }
    } else {
      console.log('\n❌ Search not returning results - API issue still exists')
    }
    
  } else {
    console.log('\n❌ Search button still not visible on Instrument Discovery tab')
  }

  console.log('\n📋 USER INSTRUCTION SUMMARY:')
  console.log('1. Go to Interactive Brokers tab')
  console.log('2. Click "Instrument Discovery" tab (NOT Order Book tab)')
  console.log('3. Click "Search Instruments" button')
  console.log('4. Enter symbol (e.g. PLTR) and click Search')
  console.log('5. Close modal to see results')
  console.log('6. Click "Order Book" button in results row')
  console.log('7. Switch to "Order Book" tab to see visualization')
})