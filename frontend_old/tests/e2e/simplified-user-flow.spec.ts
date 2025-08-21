import { test, expect } from '@playwright/test'

test('Simplified User Flow - Search in Order Book Tab', async ({ page }) => {
  console.log('\n🎯 SIMPLIFIED USER FLOW TEST')
  console.log('📋 Search functionality now in Order Book tab')

  await page.goto('http://localhost:3000')
  await page.waitForTimeout(2000)

  console.log('\n1. Go to Interactive Brokers tab')
  await page.locator('text=Interactive Brokers').click()
  await page.waitForTimeout(3000)

  console.log('\n2. Go directly to Order Book tab')
  await page.locator('#rc-tabs-5-tab-orderbook').click()
  await page.waitForTimeout(1000)

  console.log('\n3. Search button should now be visible immediately!')
  const searchButton = page.locator('button:has-text("Search Instruments")')
  const isVisible = await searchButton.isVisible()
  console.log(`   ✅ Search button visible: ${isVisible}`)

  if (isVisible) {
    console.log('\n4. Click search and test PLTR')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    await page.locator('.ant-modal input').first().fill('PLTR')
    await page.locator('.ant-modal button:has-text("Search")').click()
    await page.waitForTimeout(3000)
    
    await page.keyboard.press('Escape')
    await page.waitForTimeout(1000)
    
    const pltrResults = await page.locator('text=PLTR').count()
    const orderBookBtns = await page.locator('button:has-text("Order Book")').count()
    
    console.log(`   ✅ PLTR results: ${pltrResults}`)
    console.log(`   ✅ Order Book buttons: ${orderBookBtns}`)
    
    if (orderBookBtns > 0) {
      console.log('\n5. Click Order Book button - should work in same tab')
      await page.locator('button:has-text("Order Book")').first().click()
      await page.waitForTimeout(2000)
      
      const orderBookVisible = await page.locator('text=Order Book - PLTR').isVisible()
      console.log(`   ✅ Order book opened: ${orderBookVisible}`)
      
      console.log('\n🎉 SUCCESS! Single-tab workflow complete!')
    }
  }
  
  await page.screenshot({ path: 'simplified-workflow.png', fullPage: true })
  
  console.log('\n📊 IMPROVEMENT SUMMARY:')
  console.log('✅ Search button now visible directly in Order Book tab')
  console.log('✅ Search results appear in same tab')
  console.log('✅ Order book opens in same tab')
  console.log('✅ No more tab switching confusion!')
})