#!/usr/bin/env node

import puppeteer from 'puppeteer';

async function fixBlankPage() {
  console.log('🔍 Investigating blank page at localhost:3000...');
  
  const browser = await puppeteer.launch({ 
    headless: false,
    devtools: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Enable console logging
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    console.log(`🖥️  BROWSER ${type.toUpperCase()}: ${text}`);
  });
  
  // Track network failures
  page.on('requestfailed', request => {
    console.log(`❌ NETWORK FAILED: ${request.url()} - ${request.failure()?.errorText}`);
  });
  
  // Track JavaScript errors
  page.on('pageerror', error => {
    console.log(`💥 PAGE ERROR: ${error.message}`);
  });
  
  try {
    console.log('🌐 Navigating to localhost:3000...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'domcontentloaded',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for React to load...');
    await page.waitForTimeout(5000);
    
    // Check if root element has content
    const rootContent = await page.evaluate(() => {
      const root = document.getElementById('root');
      return {
        exists: !!root,
        innerHTML: root?.innerHTML || '',
        hasChildren: root?.children?.length || 0
      };
    });
    
    console.log('🌳 Root element analysis:', rootContent);
    
    if (rootContent.hasChildren === 0) {
      console.log('❗ Root is empty - React app not rendering!');
      
      // Check for JavaScript errors
      const jsErrors = await page.evaluate(() => {
        return window.console.errors || [];
      });
      
      console.log('🔧 Attempting to fix common issues...');
      
      // Try to manually trigger React if it's not loading
      await page.evaluate(() => {
        // Check if React is available
        if (typeof window.React !== 'undefined') {
          console.log('React is available but not rendering');
        } else {
          console.log('React is not loaded');
        }
        
        // Try to reload the main script
        const script = document.querySelector('script[src*="main.tsx"]');
        if (script) {
          console.log('Reloading main script...');
          script.remove();
          const newScript = document.createElement('script');
          newScript.type = 'module';
          newScript.src = script.getAttribute('src');
          document.head.appendChild(newScript);
        }
      });
      
      await page.waitForTimeout(3000);
      
      // Check again
      const rootContentAfter = await page.evaluate(() => {
        const root = document.getElementById('root');
        return {
          hasChildren: root?.children?.length || 0,
          innerHTML: root?.innerHTML?.substring(0, 200) || ''
        };
      });
      
      console.log('🌳 Root element after fix attempt:', rootContentAfter);
    } else {
      console.log('✅ React app appears to be rendering correctly');
    }
    
    // Keep browser open for manual inspection
    console.log('🖱️  Browser opened for manual inspection. Press Ctrl+C to close.');
    await page.waitForTimeout(60000);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
  
  await browser.close();
}

fixBlankPage().catch(console.error);