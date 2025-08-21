import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  // Listen for console messages
  page.on('console', msg => {
    console.log('BROWSER CONSOLE:', msg.type(), msg.text());
  });
  
  // Listen for page errors
  page.on('pageerror', err => {
    console.log('PAGE ERROR:', err.message);
  });
  
  try {
    console.log('Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    
    await page.waitForTimeout(3000);
    
    const title = await page.title();
    console.log('Page title:', title);
    
    const content = await page.textContent('body');
    console.log('Page content length:', content.length);
    console.log('Page content preview:', content.substring(0, 200));
    
    // Check for specific elements
    const rootElement = await page.$('#root');
    if (rootElement) {
      const rootContent = await rootElement.textContent();
      console.log('Root element content:', rootContent);
    } else {
      console.log('No #root element found');
    }
    
    // Check for React error boundary
    const errorBoundary = await page.$('[data-testid="error-boundary"]');
    if (errorBoundary) {
      console.log('Error boundary found');
    }
    
    await page.screenshot({ path: 'debug-frontend.png', fullPage: true });
    console.log('Screenshot saved as debug-frontend.png');
    
  } catch (error) {
    console.error('Error during debugging:', error);
  }
  
  await browser.close();
})();