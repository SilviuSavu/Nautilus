import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function investigateFrontend() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 1000 // Add delay to see what's happening
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  
  const page = await context.newPage();
  
  // Create screenshots directory
  const screenshotsDir = path.join(__dirname, 'investigation-screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }
  
  console.log('🔍 Starting frontend investigation...');
  
  // Track console errors
  const consoleErrors = [];
  const networkErrors = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push({
        text: msg.text(),
        location: msg.location(),
        timestamp: new Date().toISOString()
      });
      console.log('❌ Console Error:', msg.text());
    } else if (msg.type() === 'warning') {
      console.log('⚠️  Console Warning:', msg.text());
    }
  });
  
  page.on('response', response => {
    if (!response.ok() && response.status() >= 400) {
      networkErrors.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText(),
        timestamp: new Date().toISOString()
      });
      console.log('🌐 Network Error:', response.status(), response.url());
    }
  });
  
  try {
    // Step 1: Load the main page
    console.log('📱 Loading main page...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait a bit for any async loading
    await page.waitForTimeout(3000);
    
    // Take initial screenshot
    await page.screenshot({ 
      path: path.join(screenshotsDir, '01-initial-load.png'),
      fullPage: true 
    });
    
    // Step 2: Check for basic elements
    console.log('🔍 Checking for basic UI elements...');
    
    // Check if Ant Design is loading properly
    const antdElements = await page.$$('.ant-btn, .ant-menu, .ant-layout, .ant-card');
    console.log(`✅ Found ${antdElements.length} Ant Design elements`);
    
    // Check for any visible error messages
    const errorElements = await page.$$('text=/error|Error|ERROR/i');
    if (errorElements.length > 0) {
      console.log(`❌ Found ${errorElements.length} error-related elements`);
      await page.screenshot({ 
        path: path.join(screenshotsDir, '02-error-elements.png'),
        fullPage: true 
      });
    }
    
    // Step 3: Check navigation and main components
    console.log('🧭 Testing navigation...');
    
    // Look for main navigation elements
    const navItems = await page.$$('.ant-menu-item, [role="menuitem"], nav a');
    console.log(`📊 Found ${navItems.length} navigation items`);
    
    // Try to interact with dashboard elements if they exist
    const dashboardElements = await page.$$('[data-testid*="dashboard"], .dashboard, [class*="Dashboard"]');
    console.log(`📈 Found ${dashboardElements.length} dashboard-related elements`);
    
    if (dashboardElements.length > 0) {
      await page.screenshot({ 
        path: path.join(screenshotsDir, '03-dashboard-view.png'),
        fullPage: true 
      });
    }
    
    // Step 4: Check for charts and data visualization
    console.log('📊 Checking for charts and data visualizations...');
    
    const chartElements = await page.$$('canvas, svg, [class*="chart"], [class*="Chart"]');
    console.log(`📈 Found ${chartElements.length} chart elements`);
    
    if (chartElements.length > 0) {
      await page.screenshot({ 
        path: path.join(screenshotsDir, '04-charts-view.png'),
        fullPage: true 
      });
    }
    
    // Step 5: Check API connectivity by looking for data loading states
    console.log('🌐 Checking API connectivity indicators...');
    
    // Look for loading indicators
    const loadingElements = await page.$$('.ant-spin, .loading, [class*="Loading"]');
    console.log(`⏳ Found ${loadingElements.length} loading indicators`);
    
    // Look for empty states or data tables
    const dataElements = await page.$$('.ant-table, .ant-list, [class*="Table"], [class*="List"]');
    console.log(`📋 Found ${dataElements.length} data display elements`);
    
    // Step 6: Test specific trading platform features
    console.log('💹 Testing trading platform specific features...');
    
    // Look for trading-related components
    const tradingElements = await page.$$('[class*="Risk"], [class*="Portfolio"], [class*="Strategy"], [class*="Performance"]');
    console.log(`💼 Found ${tradingElements.length} trading-specific elements`);
    
    if (tradingElements.length > 0) {
      await page.screenshot({ 
        path: path.join(screenshotsDir, '05-trading-components.png'),
        fullPage: true 
      });
    }
    
    // Step 7: Check for WebSocket connections (if any indicators exist)
    console.log('🔌 Checking for real-time connection indicators...');
    
    const wsElements = await page.$$('[class*="connection"], [class*="Connection"], [class*="status"], .ant-badge');
    console.log(`🔗 Found ${wsElements.length} connection-related elements`);
    
    // Step 8: Final comprehensive screenshot
    await page.screenshot({ 
      path: path.join(screenshotsDir, '06-final-state.png'),
      fullPage: true 
    });
    
    // Step 9: Generate investigation report
    const report = {
      timestamp: new Date().toISOString(),
      pageTitle: await page.title(),
      url: page.url(),
      consoleErrors: consoleErrors,
      networkErrors: networkErrors,
      elementCounts: {
        antdElements: antdElements.length,
        errorElements: errorElements.length,
        navItems: navItems.length,
        dashboardElements: dashboardElements.length,
        chartElements: chartElements.length,
        loadingElements: loadingElements.length,
        dataElements: dataElements.length,
        tradingElements: tradingElements.length,
        wsElements: wsElements.length
      },
      screenshots: [
        '01-initial-load.png',
        '02-error-elements.png',
        '03-dashboard-view.png',
        '04-charts-view.png',
        '05-trading-components.png',
        '06-final-state.png'
      ]
    };
    
    fs.writeFileSync(
      path.join(screenshotsDir, 'investigation-report.json'), 
      JSON.stringify(report, null, 2)
    );
    
    console.log('\n📋 INVESTIGATION SUMMARY:');
    console.log('========================');
    console.log(`Page Title: ${report.pageTitle}`);
    console.log(`Console Errors: ${consoleErrors.length}`);
    console.log(`Network Errors: ${networkErrors.length}`);
    console.log(`Ant Design Elements: ${antdElements.length}`);
    console.log(`Error Elements: ${errorElements.length}`);
    console.log(`Navigation Items: ${navItems.length}`);
    console.log(`Dashboard Elements: ${dashboardElements.length}`);
    console.log(`Chart Elements: ${chartElements.length}`);
    console.log(`Trading Components: ${tradingElements.length}`);
    
    if (consoleErrors.length > 0) {
      console.log('\n❌ CONSOLE ERRORS FOUND:');
      consoleErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error.text}`);
        if (error.location) {
          console.log(`   Location: ${error.location.url}:${error.location.lineNumber}:${error.location.columnNumber}`);
        }
      });
    }
    
    if (networkErrors.length > 0) {
      console.log('\n🌐 NETWORK ERRORS FOUND:');
      networkErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error.status} ${error.statusText} - ${error.url}`);
      });
    }
    
    console.log(`\n📸 Screenshots saved to: ${screenshotsDir}`);
    console.log('📄 Full report saved to: investigation-report.json');
    
  } catch (error) {
    console.error('❌ Investigation failed:', error);
    
    // Take error screenshot
    try {
      await page.screenshot({ 
        path: path.join(screenshotsDir, 'error-state.png'),
        fullPage: true 
      });
      console.log('📸 Error state screenshot saved');
    } catch (screenshotError) {
      console.error('Failed to take error screenshot:', screenshotError);
    }
  } finally {
    await browser.close();
  }
}

investigateFrontend().catch(console.error);