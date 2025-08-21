const { chromium } = require('playwright');

async function testRiskTab() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  try {
    console.log('🔍 Going to localhost:3000...');
    await page.goto('http://localhost:3000');
    
    console.log('⏳ Waiting for dashboard to load...');
    await page.waitForSelector('[data-testid="dashboard"]', { timeout: 15000 });
    
    console.log('📸 Taking initial screenshot...');
    await page.screenshot({ path: 'frontend/containerized-initial.png' });
    
    console.log('🎯 Clicking on Risk tab...');
    await page.click('text=Risk');
    
    console.log('⏳ Waiting for Risk tab to load...');
    await page.waitForTimeout(5000);
    
    console.log('📸 Taking Risk tab screenshot...');
    await page.screenshot({ path: 'frontend/containerized-risk-final.png' });
    
    console.log('✅ Risk tab test completed successfully!');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  } finally {
    await browser.close();
  }
}

testRiskTab();