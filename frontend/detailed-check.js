import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const consoleMessages = [];
  const networkRequests = [];
  const errors = [];
  
  // Capture all console messages
  page.on('console', msg => {
    consoleMessages.push(`${msg.type()}: ${msg.text()}`);
  });
  
  // Capture network requests
  page.on('request', request => {
    if (request.url().includes('ws://') || request.url().includes('messagebus')) {
      networkRequests.push(`REQUEST: ${request.method()} ${request.url()}`);
    }
  });
  
  page.on('response', response => {
    if (response.url().includes('ws://') || response.url().includes('messagebus')) {
      networkRequests.push(`RESPONSE: ${response.status()} ${response.url()}`);
    }
  });
  
  // Capture errors
  page.on('pageerror', error => {
    errors.push(`Page Error: ${error.message}`);
  });
  
  try {
    console.log('🚀 Loading http://localhost:3000...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'domcontentloaded',
      timeout: 15000 
    });
    
    // Wait longer for WebSocket connections to attempt
    await page.waitForTimeout(5000);
    
    console.log('\n=== WEBSOCKET-RELATED CONSOLE MESSAGES ===');
    const wsMessages = consoleMessages.filter(msg => 
      msg.includes('ws://') || 
      msg.includes('WebSocket') || 
      msg.includes('messagebus') ||
      msg.includes('connection') ||
      msg.includes('3000') ||
      msg.includes('8001')
    );
    wsMessages.forEach(msg => console.log(msg));
    
    console.log('\n=== NETWORK REQUESTS ===');
    networkRequests.forEach(req => console.log(req));
    
    console.log('\n=== ERRORS ===');
    errors.forEach(error => console.log(error));
    
  } catch (error) {
    console.error('❌ Failed to load page:', error.message);
  }
  
  await browser.close();
})();