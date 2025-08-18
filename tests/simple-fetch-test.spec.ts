import { test, expect } from '@playwright/test'

test('simple fetch test', async ({ page }) => {
  // Enable all console logging
  page.on('console', msg => console.log('BROWSER:', msg.text()))
  
  // Navigate to the dashboard
  await page.goto('http://localhost:3000')
  
  // Run a simple fetch test directly in the browser
  const result = await page.evaluate(async () => {
    try {
      console.log('🧪 DIRECT FETCH TEST STARTING')
      const response = await fetch('http://localhost:8000/api/v1/ib/instruments/search/PLTR?max_results=5')
      console.log('🧪 Response status:', response.status)
      console.log('🧪 Response ok:', response.ok)
      
      if (response.ok) {
        const data = await response.json()
        console.log('🧪 Data received:', data)
        console.log('🧪 Instruments count:', data.instruments?.length || 0)
        return { success: true, count: data.instruments?.length || 0 }
      } else {
        console.log('🧪 Response not ok')
        return { success: false, status: response.status }
      }
    } catch (error) {
      console.log('🧪 FETCH ERROR:', error)
      return { success: false, error: error.message }
    }
  })
  
  console.log('Final result:', result)
  expect(result.success).toBe(true)
})