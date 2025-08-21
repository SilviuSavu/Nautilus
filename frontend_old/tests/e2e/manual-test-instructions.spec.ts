import { test, expect } from '@playwright/test'

test('Manual Test Instructions - Step by Step', async ({ page }) => {
  console.log('\n🔧 MANUAL TESTING INSTRUCTIONS')
  console.log('=' * 50)
  console.log('📋 Follow these steps to test the Order Book manually:')
  console.log('')
  console.log('1. Open Safari and go to: http://localhost:3000')
  console.log('2. Click on "Interactive Brokers" tab')
  console.log('3. Wait for IB Dashboard to load (should show "Connected")')
  console.log('4. Look for "Search Instruments" button - it should be visible')
  console.log('5. Click "Search Instruments" button')
  console.log('6. In the modal, enter "AAPL" in the Symbol field')
  console.log('7. Click the blue "Search" button')
  console.log('8. Close the modal (press Escape or click X)')
  console.log('9. You should now see AAPL in the search results table')
  console.log('10. Look for "Order Book" button in the AAPL row')
  console.log('11. Click "Order Book" button')
  console.log('12. Order Book visualization should open!')
  console.log('')
  console.log('🎯 EXPECTED RESULTS:')
  console.log('✅ Search modal opens')
  console.log('✅ AAPL search returns results')  
  console.log('✅ AAPL appears in results table')
  console.log('✅ "Order Book" button appears next to AAPL')
  console.log('✅ Order Book opens showing bid/ask levels')
  console.log('')
  console.log('🚨 IF IT DOESN\'T WORK:')
  console.log('❌ Check browser console for errors')
  console.log('❌ Verify backend is running on port 8000')
  console.log('❌ Check that IB Gateway shows "Connected"')
  
  // Keep browser open for manual testing
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(2000)
  
  // Navigate to IB tab
  await page.locator('text=Interactive Brokers').click()
  await page.waitForTimeout(3000)
  
  console.log('\n✅ Browser is ready for manual testing!')
  console.log('🔧 You can now manually test the Order Book functionality')
  console.log('⏸️  Browser will stay open for 60 seconds...')
  
  // Keep browser open for manual testing
  await page.waitForTimeout(60000)
  
  console.log('\n🎬 Manual testing time complete!')
})