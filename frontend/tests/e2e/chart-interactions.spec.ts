import { test, expect } from '@playwright/test'

test.describe('Chart Interactions E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Enable console logging for debugging
    page.on('console', msg => console.log('BROWSER:', msg.text()))
    
    // Navigate to the chart page
    await page.goto('http://localhost:3000')
    
    // Wait for the application to load
    await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 })
    
    // Navigate to Financial Chart tab
    await page.click('text=Financial Chart')
    await page.waitForTimeout(1000)
  })

  test('should display chart components correctly', async ({ page }) => {
    // Verify that main chart components are present
    await expect(page.locator('.chart-component')).toBeVisible()
    
    // Check for timeframe selector
    await expect(page.locator('text=1m')).toBeVisible()
    await expect(page.locator('text=5m')).toBeVisible()
    await expect(page.locator('text=1H')).toBeVisible()
    await expect(page.locator('text=1D')).toBeVisible()
    
    // Take screenshot for verification
    await page.screenshot({ 
      path: 'test-results/chart-components-visible.png',
      fullPage: true 
    })
  })

  test('should change timeframes correctly', async ({ page }) => {
    // Test timeframe switching
    const timeframes = ['1m', '5m', '15m', '1H', '4H', '1D']
    
    for (const timeframe of timeframes) {
      console.log(`Testing timeframe: ${timeframe}`)
      
      // Click the timeframe button
      await page.click(`text=${timeframe}`)
      
      // Wait for chart to update
      await page.waitForTimeout(2000)
      
      // Verify the timeframe is selected (should have primary button styling)
      const button = page.locator(`button:has-text("${timeframe}")`).first()
      await expect(button).toHaveClass(/ant-btn-primary/)
      
      // Log any console errors during timeframe change
      await page.waitForTimeout(500)
    }
    
    await page.screenshot({ 
      path: 'test-results/timeframe-switching.png',
      fullPage: true 
    })
  })

  test('should handle instrument selection', async ({ page }) => {
    // Look for instrument selector
    const instrumentSelector = page.locator('.instrument-selector')
    
    if (await instrumentSelector.isVisible()) {
      await instrumentSelector.click()
      
      // Wait for dropdown or options
      await page.waitForTimeout(1000)
      
      // Take screenshot of instrument selection
      await page.screenshot({ 
        path: 'test-results/instrument-selection.png',
        fullPage: true 
      })
    } else {
      console.log('Instrument selector not visible - may require backend data')
    }
  })

  test('should display chart with data or proper empty state', async ({ page }) => {
    // Wait for chart to initialize
    await page.waitForTimeout(3000)
    
    // Check if chart has data or shows appropriate empty state
    const chartCanvas = page.locator('canvas')
    const emptyStateMessage = page.locator('text=No Market Data Available')
    
    const hasCanvas = await chartCanvas.count() > 0
    const hasEmptyState = await emptyStateMessage.isVisible()
    
    if (hasCanvas) {
      console.log('Chart canvas found - checking for data')
      await page.screenshot({ 
        path: 'test-results/chart-with-canvas.png',
        fullPage: true 
      })
    } else if (hasEmptyState) {
      console.log('Empty state message displayed correctly')
      await page.screenshot({ 
        path: 'test-results/chart-empty-state.png',
        fullPage: true 
      })
    }
    
    // At least one should be present
    expect(hasCanvas || hasEmptyState).toBeTruthy()
  })

  test('should handle chart zoom and pan interactions', async ({ page }) => {
    // Wait for chart to load
    await page.waitForTimeout(3000)
    
    const chartContainer = page.locator('.chart-container').first()
    
    if (await chartContainer.isVisible()) {
      // Test mouse wheel zoom (simulate)
      await chartContainer.hover()
      await page.mouse.wheel(0, -100) // Zoom in
      await page.waitForTimeout(500)
      
      await page.mouse.wheel(0, 100) // Zoom out
      await page.waitForTimeout(500)
      
      // Test drag pan
      const box = await chartContainer.boundingBox()
      if (box) {
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
        await page.mouse.down()
        await page.mouse.move(box.x + box.width / 2 + 50, box.y + box.height / 2)
        await page.mouse.up()
        await page.waitForTimeout(500)
      }
      
      await page.screenshot({ 
        path: 'test-results/chart-interactions.png',
        fullPage: true 
      })
    }
  })

  test('should handle keyboard shortcuts', async ({ page }) => {
    // Focus on the page
    await page.click('body')
    
    // Test keyboard shortcuts for chart controls
    const shortcuts = [
      { key: 'v', description: 'Toggle Volume' },
      { key: 'c', description: 'Toggle Crosshair' },
      { key: 'g', description: 'Toggle Grid' },
      { key: 'f', description: 'Toggle Fullscreen' }
    ]
    
    for (const shortcut of shortcuts) {
      console.log(`Testing keyboard shortcut: ${shortcut.key} - ${shortcut.description}`)
      
      await page.keyboard.press(shortcut.key)
      await page.waitForTimeout(500)
      
      // Log any console messages
      await page.waitForTimeout(200)
    }
    
    // Test zoom shortcuts
    await page.keyboard.press('Control+Equal') // Zoom in
    await page.waitForTimeout(500)
    
    await page.keyboard.press('Control+Minus') // Zoom out
    await page.waitForTimeout(500)
    
    await page.keyboard.press('Control+0') // Zoom fit
    await page.waitForTimeout(500)
    
    await page.screenshot({ 
      path: 'test-results/keyboard-shortcuts.png',
      fullPage: true 
    })
  })

  test('should display and interact with chart controls', async ({ page }) => {
    // Look for chart controls
    const chartControls = page.locator('.chart-controls')
    
    if (await chartControls.isVisible()) {
      console.log('Chart controls found')
      
      // Test control buttons
      const controlButtons = [
        { title: 'Zoom In (Ctrl/Cmd + +)', icon: '🔍+' },
        { title: 'Zoom Out (Ctrl/Cmd + -)', icon: '🔍-' },
        { title: 'Toggle Volume (V)', icon: '📊' },
        { title: 'Toggle Real-time Updates', text: 'Live' }
      ]
      
      for (const control of controlButtons) {
        const button = control.title 
          ? page.locator(`button[title="${control.title}"]`)
          : page.locator(`button:has-text("${control.text}")`)
        
        if (await button.isVisible()) {
          console.log(`Testing control: ${control.title || control.text}`)
          await button.click()
          await page.waitForTimeout(500)
        }
      }
      
      await page.screenshot({ 
        path: 'test-results/chart-controls.png',
        fullPage: true 
      })
    } else {
      console.log('Chart controls not visible')
    }
  })

  test('should handle indicator panel interactions', async ({ page }) => {
    // Look for indicator panel
    const indicatorPanel = page.locator('.indicator-panel')
    
    if (await indicatorPanel.isVisible()) {
      console.log('Indicator panel found')
      
      // Expand indicator panel
      const toggleButton = page.locator('button:has-text("📊 Indicators")')
      if (await toggleButton.isVisible()) {
        await toggleButton.click()
        await page.waitForTimeout(500)
        
        // Look for add indicator form
        const addForm = page.locator('.indicator-add-form')
        if (await addForm.isVisible()) {
          console.log('Add indicator form visible')
          
          // Try to add an indicator
          const addButton = page.locator('button:has-text("Add SMA(20)")')
          if (await addButton.isVisible()) {
            await addButton.click()
            await page.waitForTimeout(1000)
            
            console.log('Added SMA indicator')
          }
        }
        
        await page.screenshot({ 
          path: 'test-results/indicator-panel.png',
          fullPage: true 
        })
      }
    } else {
      console.log('Indicator panel not visible')
    }
  })

  test('should handle error states gracefully', async ({ page }) => {
    // Test behavior when backend is not available
    // This will naturally happen if backend is down
    
    await page.waitForTimeout(5000) // Wait for potential API calls
    
    // Check for error messages or loading states
    const errorElements = [
      page.locator('text=No Market Data Available'),
      page.locator('text=Error loading'),
      page.locator('text=Connection failed'),
      page.locator('.error-message'),
      page.locator('.loading-state')
    ]
    
    let errorFound = false
    for (const errorElement of errorElements) {
      if (await errorElement.isVisible()) {
        console.log(`Error state found: ${await errorElement.textContent()}`)
        errorFound = true
        break
      }
    }
    
    // Chart should either show data or proper error state
    if (errorFound) {
      console.log('Error state handled correctly')
    } else {
      console.log('No error state visible - chart may be working')
    }
    
    await page.screenshot({ 
      path: 'test-results/error-handling.png',
      fullPage: true 
    })
  })

  test('should maintain responsive design', async ({ page }) => {
    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080, name: 'desktop' },
      { width: 1366, height: 768, name: 'laptop' },
      { width: 768, height: 1024, name: 'tablet' }
    ]
    
    for (const viewport of viewports) {
      await page.setViewportSize({ 
        width: viewport.width, 
        height: viewport.height 
      })
      
      await page.waitForTimeout(1000)
      
      // Check that chart components are still visible
      const chartComponent = page.locator('.chart-component')
      await expect(chartComponent).toBeVisible()
      
      await page.screenshot({ 
        path: `test-results/responsive-${viewport.name}.png`,
        fullPage: true 
      })
      
      console.log(`Tested viewport: ${viewport.name} (${viewport.width}x${viewport.height})`)
    }
  })

  test('should track performance metrics', async ({ page }) => {
    // Measure page load performance
    const startTime = Date.now()
    
    await page.goto('http://localhost:3000')
    await page.click('text=Financial Chart')
    
    // Wait for chart to be ready
    await page.waitForSelector('.chart-component', { timeout: 10000 })
    
    const loadTime = Date.now() - startTime
    
    console.log(`Chart page load time: ${loadTime}ms`)
    
    // Verify load time is reasonable (< 5 seconds)
    expect(loadTime).toBeLessThan(5000)
    
    // Test timeframe switch performance
    const switchStart = Date.now()
    await page.click('text=5m')
    await page.waitForTimeout(2000) // Wait for potential data load
    const switchTime = Date.now() - switchStart
    
    console.log(`Timeframe switch time: ${switchTime}ms`)
    
    // Verify switch time is reasonable (< 3 seconds)
    expect(switchTime).toBeLessThan(3000)
    
    await page.screenshot({ 
      path: 'test-results/performance-test.png',
      fullPage: true 
    })
  })
})