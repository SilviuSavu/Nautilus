import { chromium } from 'playwright';

(async () => {
  console.log('🚀 Starting browser to show you the frontend...');
  
  const browser = await chromium.launch({ 
    headless: false, // Show the browser
    slowMo: 1000     // Slow down actions so you can see them
  });
  
  const page = await browser.newPage();
  
  // Make the browser window larger
  await page.setViewportSize({ width: 1920, height: 1080 });
  
  console.log('🌐 Navigating to the Nautilus Trader Frontend...');
  await page.goto('http://localhost:3000/dashboard');
  
  console.log('⏳ Waiting for page to load...');
  await page.waitForTimeout(3000);
  
  console.log('📸 Taking homepage screenshot...');
  await page.screenshot({ path: 'homepage-demo.png', fullPage: true });
  
  console.log('✅ Checking for dashboard...');
  try {
    await page.waitForSelector('[data-testid="dashboard"]', { timeout: 5000 });
    console.log('✅ Dashboard found and loaded!');
  } catch (e) {
    console.log('⚠️ Dashboard element not found, but page loaded');
  }
  
  console.log('🏷️ Looking for tabs...');
  await page.waitForTimeout(2000);
  
  console.log('📊 Clicking on Financial Chart tab...');
  try {
    await page.click('text=Financial Chart');
    await page.waitForTimeout(3000);
    console.log('📈 Financial Chart tab opened!');
    await page.screenshot({ path: 'chart-tab-demo.png', fullPage: true });
  } catch (e) {
    console.log('⚠️ Could not click Financial Chart tab');
  }
  
  console.log('🔍 Looking for Instrument Search tab...');
  try {
    await page.click('text=Instrument Search');
    await page.waitForTimeout(2000);
    console.log('🔍 Instrument Search tab opened!');
    await page.screenshot({ path: 'search-tab-demo.png', fullPage: true });
  } catch (e) {
    console.log('⚠️ Could not click Instrument Search tab');
  }
  
  console.log('🏠 Going back to System Overview...');
  try {
    await page.click('text=System Overview');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'system-overview-demo.png', fullPage: true });
  } catch (e) {
    console.log('⚠️ Could not click System Overview tab');
  }
  
  console.log('🎉 Frontend demo complete!');
  console.log('📸 Screenshots saved:');
  console.log('   - homepage-demo.png');
  console.log('   - chart-tab-demo.png');  
  console.log('   - search-tab-demo.png');
  console.log('   - system-overview-demo.png');
  
  console.log('⏰ Browser will stay open for 60 seconds so you can explore...');
  console.log('💡 You can manually interact with the frontend now!');
  console.log('🌐 URL: http://localhost:3000/dashboard');
  
  await page.waitForTimeout(60000);
  
  await browser.close();
  console.log('✅ Demo complete!');
})();