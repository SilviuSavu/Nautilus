import { chromium } from 'playwright';

(async () => {
  console.log('🚀 Testing dashboard with error boundaries...');
  
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  // Listen for console logs
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    if (type === 'error') {
      console.log('❌ BROWSER ERROR:', text);
    } else if (type === 'warn') {
      console.log('⚠️  BROWSER WARN:', text);
    } else if (text.includes('🚀') || text.includes('✅') || text.includes('❌')) {
      console.log('📝 BROWSER LOG:', text);
    }
  });
  
  try {
    console.log('🌐 Navigating to dashboard...');
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'networkidle' });
    
    console.log('⏱️  Waiting for dashboard to load...');
    await page.waitForTimeout(5000);
    
    // Check if dashboard element exists
    const dashboardElement = await page.locator('[data-testid="dashboard"]').count();
    console.log(`📊 Dashboard element found: ${dashboardElement > 0 ? 'YES' : 'NO'}`);
    
    // Check if tabs are visible
    const tabsCount = await page.locator('.ant-tabs-tab').count();
    console.log(`📑 Tabs visible: ${tabsCount}`);
    
    // Check for error boundaries
    const errorBoundaries = await page.locator('.ant-alert-error').count();
    console.log(`🛡️  Error boundaries triggered: ${errorBoundaries}`);
    
    // Check if title is visible
    const titleVisible = await page.locator('h2:has-text("NautilusTrader Dashboard")').count();
    console.log(`📝 Title visible: ${titleVisible > 0 ? 'YES' : 'NO'}`);
    
    // Take screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ path: 'dashboard-fixed.png', fullPage: true });
    
    // Test each tab
    const tabs = ['system', 'instruments', 'watchlists', 'chart', 'ib'];
    for (const tab of tabs) {
      try {
        console.log(`🔍 Testing ${tab} tab...`);
        await page.click(`[data-node-key="${tab}"]`);
        await page.waitForTimeout(2000);
        
        const tabContent = await page.locator('.ant-tabs-tabpane-active').count();
        console.log(`  ✅ ${tab} tab loaded: ${tabContent > 0 ? 'YES' : 'NO'}`);
      } catch (error) {
        console.log(`  ❌ ${tab} tab failed: ${error.message}`);
      }
    }
    
    console.log('✅ Dashboard test complete!');
    console.log('📄 Screenshot saved as: dashboard-fixed.png');
    
  } catch (error) {
    console.log('❌ Test failed:', error.message);
    await page.screenshot({ path: 'dashboard-error.png', fullPage: true });
  }
  
  await browser.close();
})();