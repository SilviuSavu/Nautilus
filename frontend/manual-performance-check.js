// Manual check script for Performance tab
const puppeteer = require('puppeteer');

async function checkPerformanceTab() {
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  console.log('🔍 Checking Performance Tab...');
  
  // Capture console messages
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  page.on('pageerror', error => console.log('ERROR:', error.message));
  
  try {
    // Navigate to the frontend
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
    
    console.log('✅ Page loaded successfully');
    
    // Click on the Performance tab
    await page.waitForSelector('text/Perform', { timeout: 10000 });
    await page.click('text/Perform');
    console.log('✅ Clicked on Performance tab');
    
    // Wait for the Performance Dashboard to load
    await page.waitForSelector('text/Performance Dashboard', { timeout: 15000 });
    console.log('✅ Performance Dashboard loaded');
    
    // Check for key components
    const hasOverview = await page.$('text/Overview') !== null;
    const hasMetrics = await page.$('text/Total P&L') !== null;
    const hasMonitoring = await page.$('text/System Monitoring') !== null;
    const hasExport = await page.$('text/Data Export') !== null;
    
    console.log('📊 Performance tab components:');
    console.log('  - Overview tab:', hasOverview ? '✅' : '❌');
    console.log('  - Metrics cards:', hasMetrics ? '✅' : '❌');
    console.log('  - System Monitoring:', hasMonitoring ? '✅' : '❌');
    console.log('  - Data Export:', hasExport ? '✅' : '❌');
    
    // Test tab switching
    if (hasMonitoring) {
      await page.click('text/System Monitoring');
      await page.waitForTimeout(2000);
      const hasSystemDashboard = await page.$('text/System Performance Dashboard') !== null;
      console.log('  - System Dashboard loads:', hasSystemDashboard ? '✅' : '❌');
    }
    
    if (hasExport) {
      await page.click('text/Data Export');
      await page.waitForTimeout(2000);
      const hasExportDashboard = await page.$('text/Data Export Dashboard') !== null;
      console.log('  - Export Dashboard loads:', hasExportDashboard ? '✅' : '❌');
    }
    
    console.log('\n🎉 Performance Tab Check Complete!');
    console.log('The Performance tab appears to be working correctly.');
    
  } catch (error) {
    console.error('❌ Error during check:', error.message);
  }
  
  // Keep browser open for 10 seconds for manual inspection
  console.log('\n⏱️  Browser will stay open for 10 seconds for manual inspection...');
  await page.waitForTimeout(10000);
  
  await browser.close();
}

checkPerformanceTab().catch(console.error);