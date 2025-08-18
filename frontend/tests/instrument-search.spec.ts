import { test, expect } from '@playwright/test';

test('Instrument Search Investigation', async ({ page }) => {
  // Capture all console messages
  const consoleMessages: string[] = [];
  page.on('console', msg => {
    const message = `[${msg.type()}] ${msg.text()}`;
    consoleMessages.push(message);
    console.log('BROWSER:', message);
  });

  // Capture network requests
  const apiCalls: { method: string, url: string, status?: number }[] = [];
  page.on('request', request => {
    if (request.url().includes('localhost:8000')) {
      apiCalls.push({ method: request.method(), url: request.url() });
      console.log('API REQUEST:', `${request.method()} ${request.url()}`);
    }
  });

  page.on('response', response => {
    if (response.url().includes('localhost:8000')) {
      const call = apiCalls.find(c => c.url === response.url() && !c.status);
      if (call) {
        call.status = response.status();
        console.log('API RESPONSE:', `${response.status()} ${response.url()}`);
      }
    }
  });

  console.log('=== NAVIGATING TO APPLICATION ===');
  await page.goto('http://localhost:3000');
  
  // Take initial screenshot
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/instrument-search-initial.png', fullPage: true });
  
  // Wait for the page to load
  await page.waitForTimeout(3000);
  
  console.log('=== LOOKING FOR INSTRUMENT SEARCH ELEMENTS ===');
  
  // Look for search inputs/fields
  const searchInputs = await page.locator('input[type="text"], input[type="search"], input[placeholder*="search"], input[placeholder*="symbol"], input[placeholder*="instrument"], [class*="search"], [data-testid*="search"]').count();
  console.log(`Search input elements found: ${searchInputs}`);
  
  // Look for any search-related buttons
  const searchButtons = await page.locator('button:has-text("Search"), button[type="submit"], [class*="search-btn"], [data-testid*="search-btn"]').count();
  console.log(`Search button elements found: ${searchButtons}`);
  
  // Look for dropdowns or autocomplete
  const dropdowns = await page.locator('select, [class*="dropdown"], [class*="autocomplete"], [role="combobox"], [role="listbox"]').count();
  console.log(`Dropdown/autocomplete elements found: ${dropdowns}`);
  
  // Take screenshot of current state
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/instrument-search-elements.png', fullPage: true });
  
  // Try to find and interact with search functionality
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
    console.log('=== TESTING INSTRUMENT SEARCH FUNCTIONALITY ===');
    
    // Test searching for common symbols
    const testSymbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA'];
    
    for (const symbol of testSymbols) {
      console.log(`Testing search for: ${symbol}`);
      
      // Clear and type the symbol
      await searchInput.clear();
      await searchInput.fill(symbol);
      
      // Wait for any autocomplete or search results
      await page.waitForTimeout(1500);
      
      // Take screenshot
      await page.screenshot({ 
        path: `/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/search-${symbol}.png`, 
        fullPage: true 
      });
      
      // Look for search results
      const results = await page.locator('[class*="result"], [class*="option"], [role="option"], li, .dropdown-item').count();
      console.log(`Search results for ${symbol}: ${results}`);
      
      // Check if Enter key triggers anything
      await searchInput.press('Enter');
      await page.waitForTimeout(1000);
      
      // Take screenshot after Enter
      await page.screenshot({ 
        path: `/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/search-${symbol}-after-enter.png`, 
        fullPage: true 
      });
    }
  } else {
    console.log('=== NO SEARCH INPUT FOUND - CHECKING PAGE CONTENT ===');
    
    // Get page content to understand what's there
    const pageTitle = await page.title();
    console.log(`Page title: ${pageTitle}`);
    
    const headings = await page.locator('h1, h2, h3, h4').allTextContents();
    console.log('Page headings:', headings);
    
    const buttons = await page.locator('button').allTextContents();
    console.log('Buttons on page:', buttons.slice(0, 10)); // First 10 buttons
    
    const inputs = await page.locator('input').count();
    console.log(`Total input elements: ${inputs}`);
    
    if (inputs > 0) {
      const inputTypes = await page.locator('input').evaluateAll(inputs => 
        inputs.map(input => ({ 
          type: input.type, 
          placeholder: input.placeholder, 
          className: input.className 
        }))
      );
      console.log('Input details:', inputTypes);
    }
  }
  
  // Test direct API calls to understand backend functionality
  console.log('=== TESTING BACKEND SEARCH ENDPOINTS ===');
  
  const searchEndpoints = [
    '/symbols',
    '/search?query=AAPL',
    '/instruments',
    '/instruments/search?query=AAPL',
    '/symbols/search?query=AAPL'
  ];
  
  for (const endpoint of searchEndpoints) {
    try {
      const response = await page.request.get(`http://localhost:8000${endpoint}`);
      console.log(`${endpoint}: ${response.status()}`);
      
      if (response.ok()) {
        const data = await response.json();
        console.log(`${endpoint} response:`, JSON.stringify(data, null, 2).substring(0, 500));
      } else {
        const errorText = await response.text();
        console.log(`${endpoint} error:`, errorText);
      }
    } catch (e) {
      console.log(`${endpoint} failed:`, e.message);
    }
  }
  
  // Final summary
  console.log('=== INSTRUMENT SEARCH INVESTIGATION SUMMARY ===');
  console.log(`Search inputs found: ${searchInputs}`);
  console.log(`Search buttons found: ${searchButtons}`);
  console.log(`Dropdowns found: ${dropdowns}`);
  console.log(`API calls made: ${apiCalls.length}`);
  
  console.log('\n=== API CALLS SUMMARY ===');
  apiCalls.forEach(call => console.log(`${call.method} ${call.url} -> ${call.status || 'pending'}`));
  
  // Take final screenshot
  await page.screenshot({ path: '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/tests/screenshots/instrument-search-final.png', fullPage: true });
});