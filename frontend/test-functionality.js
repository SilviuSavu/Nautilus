import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testApplicationFunctionality() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 1000
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  const screenshotsDir = path.join(__dirname, 'functionality-screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }
  
  console.log('🧪 Testing application functionality...');
  
  try {
    // Step 1: Load the application
    console.log('📱 Loading application...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(screenshotsDir, '01-app-loaded.png') });
    
    // Step 2: Test Backend Connection
    console.log('🔌 Testing backend connection...');
    const testButton = await page.locator('text=Test Backend Connection');
    if (await testButton.isVisible()) {
      await testButton.click();
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(screenshotsDir, '02-backend-test.png') });
      console.log('✅ Backend connection test button clicked');
    } else {
      console.log('⚠️  Backend test button not found');
    }
    
    // Step 3: Check for navigation menu
    console.log('🧭 Checking for navigation...');
    const menuElements = await page.$$('[role="menuitem"], .ant-menu-item, nav a, [class*="nav"]');
    console.log(`📊 Found ${menuElements.length} navigation elements`);
    
    // Step 4: Look for any dashboard content that might load
    console.log('📈 Waiting for dashboard content...');
    await page.waitForTimeout(5000);
    
    // Check for any dynamically loaded content
    const cards = await page.$$('.ant-card');
    const buttons = await page.$$('.ant-btn');
    const loading = await page.$$('.ant-spin');
    
    console.log(`📋 Found ${cards.length} cards, ${buttons.length} buttons, ${loading.length} loading indicators`);
    
    await page.screenshot({ path: path.join(screenshotsDir, '03-dashboard-content.png') });
    
    // Step 5: Test any clickable elements
    console.log('🖱️  Testing interactive elements...');
    
    // Try clicking "View System Status" if available
    const statusButton = await page.locator('text=View System Status');
    if (await statusButton.isVisible()) {
      await statusButton.click();
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(screenshotsDir, '04-system-status.png') });
      console.log('✅ System status button clicked');
    }
    
    // Step 6: Check browser network tab for any errors
    console.log('🌐 Checking for network activity...');
    const networkRequests = [];
    page.on('response', response => {
      networkRequests.push({
        url: response.url(),
        status: response.status(),
        ok: response.ok()
      });
    });
    
    // Trigger some actions to generate network requests
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);
    
    await page.screenshot({ path: path.join(screenshotsDir, '05-after-reload.png') });
    
    // Step 7: Final comprehensive screenshot
    await page.screenshot({ 
      path: path.join(screenshotsDir, '06-final-functionality-test.png'),
      fullPage: true 
    });
    
    // Generate report
    const report = {
      timestamp: new Date().toISOString(),
      pageTitle: await page.title(),
      url: page.url(),
      elementCounts: {
        cards: cards.length,
        buttons: buttons.length,
        loading: loading.length,
        menuElements: menuElements.length
      },
      networkRequests: networkRequests.slice(-10), // Last 10 requests
      testResults: {
        applicationLoaded: true,
        backendTestButtonFound: await page.locator('text=Test Backend Connection').isVisible(),
        systemStatusButtonFound: await page.locator('text=View System Status').isVisible(),
        antdElementsPresent: cards.length > 0 || buttons.length > 0
      }
    };
    
    fs.writeFileSync(
      path.join(screenshotsDir, 'functionality-report.json'), 
      JSON.stringify(report, null, 2)
    );
    
    console.log('\n🎉 FUNCTIONALITY TEST RESULTS:');
    console.log('================================');
    console.log(`✅ Application loaded successfully: ${report.testResults.applicationLoaded}`);
    console.log(`✅ Backend test button available: ${report.testResults.backendTestButtonFound}`);
    console.log(`✅ System status button available: ${report.testResults.systemStatusButtonFound}`);
    console.log(`✅ Ant Design elements present: ${report.testResults.antdElementsPresent}`);
    console.log(`📊 UI Elements found: ${report.elementCounts.cards} cards, ${report.elementCounts.buttons} buttons`);
    console.log(`🌐 Network requests captured: ${report.networkRequests.length}`);
    console.log(`📸 Screenshots saved to: ${screenshotsDir}`);
    
  } catch (error) {
    console.error('❌ Functionality test failed:', error);
    await page.screenshot({ 
      path: path.join(screenshotsDir, 'error-functionality-test.png'),
      fullPage: true 
    });
  } finally {
    await browser.close();
  }
}

testApplicationFunctionality().catch(console.error);