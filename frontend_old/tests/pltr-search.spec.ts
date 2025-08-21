import { test, expect } from '@playwright/test';

test('PLTR Instrument Search Investigation', async ({ page }) => {
  // Capture all console messages
  const consoleMessages: string[] = [];
  page.on('console', msg => {
    const message = `[${msg.type()}] ${msg.text()}`;
    consoleMessages.push(message);
    console.log('BROWSER:', message);
  });

  // Capture network requests/responses
  const apiCalls: { method: string, url: string, status?: number, response?: any }[] = [];
  page.on('request', request => {
    if (request.url().includes('localhost:8000')) {
      apiCalls.push({ method: request.method(), url: request.url() });
      console.log('API REQUEST:', `${request.method()} ${request.url()}`);
    }
  });

  page.on('response', async response => {
    if (response.url().includes('localhost:8000')) {
      const call = apiCalls.find(c => c.url === response.url() && !c.status);
      if (call) {
        call.status = response.status();
        try {
          if (response.headers()['content-type']?.includes('application/json')) {
            call.response = await response.json();
          }
        } catch (e) {
          console.log('Could not parse response as JSON');
        }
        console.log('API RESPONSE:', `${response.status()} ${response.url()}`);
      }
    }
  });

  console.log('=== TESTING PLTR INSTRUMENT SEARCH ===');
  await page.goto('http://localhost:3000');
  
  // Take initial screenshot
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/pltr-initial.png', fullPage: true });
  
  // Wait for page load
  await page.waitForTimeout(3000);
  
  // Test backend endpoint directly first
  console.log('=== TESTING PLTR BACKEND ENDPOINT DIRECTLY ===');
  try {
    const response = await page.request.get('http://localhost:8000/api/v1/ib/instruments/search/PLTR?max_results=100');
    console.log(`PLTR search endpoint status: ${response.status()}`);
    
    if (response.ok()) {
      const data = await response.json();
      console.log('PLTR search response:', JSON.stringify(data, null, 2));
    } else {
      const errorText = await response.text();
      console.log('PLTR search error:', errorText);
    }
  } catch (e) {
    console.log('PLTR backend search failed:', e.message);
  }
  
  // Find search input
  const searchSelectors = [
    'input[placeholder*="search" i]',
    'input[placeholder*="symbol" i]', 
    'input[placeholder*="instrument" i]',
    'input[type="search"]',
    'input[type="text"]',
    '[data-testid*="search"]',
    '[class*="search-input"]'
  ];
  
  let searchInput = null;
  for (const selector of searchSelectors) {
    const element = page.locator(selector).first();
    if (await element.isVisible()) {
      searchInput = element;
      console.log(`Found search input with selector: ${selector}`);
      break;
    }
  }
  
  if (searchInput) {
    console.log('=== TESTING PLTR SEARCH IN UI ===');
    
    // Clear and search for PLTR
    await searchInput.clear();
    await searchInput.fill('PLTR');
    
    // Take screenshot after typing
    await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/pltr-typed.png', fullPage: true });
    
    // Wait for search results
    await page.waitForTimeout(3000);
    
    // Take screenshot after waiting
    await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/pltr-after-wait.png', fullPage: true });
    
    // Count search results
    const resultSelectors = [
      '[class*="ant-list-item"]',
      '[class*="result"]',
      '[class*="option"]',
      '[role="option"]',
      'li',
      '.dropdown-item'
    ];
    
    let resultsCount = 0;
    for (const selector of resultSelectors) {
      const count = await page.locator(selector).count();
      if (count > 0) {
        resultsCount = count;
        console.log(`Found ${count} results with selector: ${selector}`);
        break;
      }
    }
    
    console.log(`Total PLTR search results found: ${resultsCount}`);
    
    // Press Enter to execute search
    await searchInput.press('Enter');
    await page.waitForTimeout(2000);
    
    // Take final screenshot
    await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/pltr-after-enter.png', fullPage: true });
    
    // Check for any result items again after Enter
    resultsCount = 0;
    for (const selector of resultSelectors) {
      const count = await page.locator(selector).count();
      if (count > resultsCount) {
        resultsCount = count;
        console.log(`After Enter - Found ${count} results with selector: ${selector}`);
      }
    }
    
    console.log(`Final PLTR search results count: ${resultsCount}`);
  } else {
    console.log('=== NO SEARCH INPUT FOUND ===');
    
    // Get available inputs for debugging
    const allInputs = await page.locator('input').count();
    console.log(`Total inputs on page: ${allInputs}`);
    
    if (allInputs > 0) {
      const inputDetails = await page.locator('input').evaluateAll(inputs => 
        inputs.map((input, index) => ({ 
          index,
          type: input.type, 
          placeholder: input.placeholder, 
          className: input.className,
          visible: input.offsetParent !== null
        }))
      );
      console.log('All input details:', inputDetails);
    }
  }
  
  // Final summary
  console.log('=== PLTR SEARCH INVESTIGATION SUMMARY ===');
  console.log(`API calls made: ${apiCalls.length}`);
  console.log(`Console messages: ${consoleMessages.length}`);
  
  console.log('\n=== API CALLS FOR PLTR ===');
  apiCalls.forEach(call => {
    console.log(`${call.method} ${call.url} -> ${call.status || 'pending'}`);
    if (call.response && call.url.includes('PLTR')) {
      console.log('Response:', JSON.stringify(call.response, null, 2));
    }
  });
  
  console.log('\n=== RECENT CONSOLE MESSAGES ===');
  consoleMessages.slice(-10).forEach(msg => console.log(msg));
});