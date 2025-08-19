import { test, expect } from '@playwright/test'

test('Real User Experience - Safari Instrument Selection', async ({ page }) => {
  page.on('console', msg => console.log(`🌐 SAFARI: ${msg.text()}`))
  
  console.log('\n🌐 REAL USER TESTING IN SAFARI')
  console.log('📋 Showing exactly what happens when trying to select instruments')
  console.log('=' * 60)

  console.log('\n🚀 Loading application...')
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(3000)

  console.log('\n🎯 Navigating to Interactive Brokers...')
  const ibTab = page.locator('text=Interactive Brokers').first()
  await ibTab.click()
  await page.waitForTimeout(4000)

  console.log('\n🔍 What does a real user see? Scanning the page...')
  
  // Show exactly what's visible
  const buttons = await page.locator('button:visible').allTextContents()
  console.log('\n👀 VISIBLE BUTTONS:')
  buttons.forEach((btn, i) => {
    if (btn.trim()) console.log(`   ${i+1}. "${btn.trim()}"`)
  })

  const inputs = await page.locator('input:visible').count()
  console.log(`\n📝 VISIBLE INPUT FIELDS: ${inputs}`)
  
  if (inputs > 0) {
    for (let i = 0; i < Math.min(inputs, 5); i++) {
      const input = page.locator('input:visible').nth(i)
      const placeholder = await input.getAttribute('placeholder')
      const type = await input.getAttribute('type')
      console.log(`   ${i+1}. Type: ${type}, Placeholder: "${placeholder || 'none'}"`)
    }
  }

  console.log('\n🔎 LOOKING FOR SEARCH FUNCTIONALITY...')
  
  // Check for search options
  const searchElements = [
    { name: 'Search Button', selector: 'button:has-text("Search")' },
    { name: 'Search Instruments', selector: 'button:has-text("Search Instruments")' },
    { name: 'Symbol Input', selector: 'input[placeholder*="symbol" i]' },
    { name: 'Any Text Input', selector: 'input[type="text"]:visible' }
  ]
  
  let canSearch = false
  for (const element of searchElements) {
    const count = await page.locator(element.selector).count()
    const visible = count > 0 ? await page.locator(element.selector).first().isVisible() : false
    
    console.log(`   ${element.name}: ${visible ? '✅ FOUND' : '❌ NOT FOUND'} (${count} total)`)
    
    if (visible && !canSearch) {
      console.log(`   🎯 ATTEMPTING TO USE: ${element.name}`)
      try {
        const el = page.locator(element.selector).first()
        
        if (element.name.includes('Input')) {
          await el.click()
          await el.fill('AAPL')
          console.log('   📝 Typed AAPL')
          canSearch = true
        } else {
          await el.click()
          console.log('   🖱️  Clicked button')
          await page.waitForTimeout(2000)
          
          // Check if modal opened
          const modal = await page.locator('.ant-modal, .modal').count()
          if (modal > 0) {
            console.log('   📋 Modal opened, looking for symbol input...')
            const modalInput = page.locator('.ant-modal input, .modal input').first()
            if (await modalInput.isVisible()) {
              await modalInput.fill('AAPL')
              console.log('   📝 Typed AAPL in modal')
              
              const searchBtn = page.locator('.ant-modal button:has-text("Search"), .modal button:has-text("Search")').first()
              if (await searchBtn.isVisible()) {
                await searchBtn.click()
                console.log('   🔍 Clicked search in modal')
                await page.waitForTimeout(3000)
              }
            }
          }
          canSearch = true
        }
        await page.waitForTimeout(2000)
      } catch (e) {
        console.log(`   ⚠️  Failed to use ${element.name}: ${e}`)
      }
    }
  }

  if (!canSearch) {
    console.log('\n🚨 CRITICAL ISSUE: NO WAY TO SEARCH FOR INSTRUMENTS!')
    console.log('   User cannot enter AAPL or any symbol')
    console.log('   No search functionality visible or accessible')
  }

  console.log('\n📊 LOOKING FOR INSTRUMENT RESULTS...')
  
  // Check for any AAPL or Apple mentions
  const aaplMentions = await page.locator('text=AAPL').count()
  const appleMentions = await page.locator('text=Apple').count()
  const tableRows = await page.locator('table tr').count()
  
  console.log(`   AAPL mentions: ${aaplMentions}`)
  console.log(`   Apple mentions: ${appleMentions}`)
  console.log(`   Table rows: ${tableRows}`)
  
  if (aaplMentions === 0 && appleMentions === 0) {
    console.log('   🚨 NO INSTRUMENT DATA VISIBLE TO USER')
  }

  console.log('\n📋 ORDER BOOK ACCESS CHECK...')
  
  const orderBookMentions = await page.locator('text=Order Book').count()
  console.log(`   "Order Book" text found: ${orderBookMentions} times`)
  
  if (orderBookMentions > 0) {
    const orderBookButtons = await page.locator('button:has-text("Order Book")').count()
    console.log(`   Clickable Order Book buttons: ${orderBookButtons}`)
    
    if (orderBookButtons === 0) {
      console.log('   ⚠️  Order Book text exists but no clickable buttons')
    }
  } else {
    console.log('   ❌ No Order Book access visible')
  }

  console.log('\n📸 Taking final screenshot...')
  await page.screenshot({ path: 'safari-real-user-experience.png', fullPage: true })

  console.log('\n🎯 REAL USER EXPERIENCE SUMMARY:')
  console.log('=' * 50)
  
  if (!canSearch) {
    console.log('❌ BLOCKING ISSUE: Cannot search for instruments')
    console.log('   - User has no way to enter stock symbols')
    console.log('   - No visible search interface')
    console.log('   - Order Book cannot be accessed without instrument selection')
    console.log('')
    console.log('🔧 REQUIRED FIX: Add visible instrument search UI')
  } else {
    console.log('✅ Search functionality exists')
  }
  
  if (aaplMentions === 0 && appleMentions === 0) {
    console.log('❌ BLOCKING ISSUE: No instrument results shown')
    console.log('   - Search may work but no results displayed')
    console.log('   - User cannot select any instruments')
  }
  
  console.log('\n💡 MISSING: Clear user journey from search to Order Book')
  console.log('📸 Screenshot saved: safari-real-user-experience.png')
})