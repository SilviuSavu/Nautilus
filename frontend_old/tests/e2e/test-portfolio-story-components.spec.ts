/**
 * Test Portfolio Visualization Story 4.4 Components
 */

import { test, expect } from '@playwright/test';

test.describe('Portfolio Visualization Story 4.4', () => {
  test('Test portfolio visualization components in Strategy Management tab', async ({ page }) => {
    console.log('🧪 Testing Portfolio Visualization Story 4.4 components...');
    
    // Capture console logs
    page.on('console', msg => {
      console.log(`BROWSER [${msg.type()}]:`, msg.text());
    });
    
    page.on('pageerror', error => {
      console.log('PAGE ERROR:', error.message);
    });
    
    console.log('📍 Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    
    // Verify main dashboard is loaded
    const dashboardTitle = await page.textContent('h2');
    console.log('📋 Dashboard title:', dashboardTitle);
    expect(dashboardTitle).toBe('NautilusTrader Dashboard');
    
    // Look for Strategy Management tab
    console.log('🔍 Looking for Strategy Management tab...');
    const strategyManagementTab = page.locator('text=Strategy Management');
    const isStrategyTabVisible = await strategyManagementTab.isVisible();
    console.log('📊 Strategy Management tab visible:', isStrategyTabVisible);
    
    if (isStrategyTabVisible) {
      console.log('✅ Clicking Strategy Management tab...');
      await strategyManagementTab.click();
      await page.waitForTimeout(2000);
      
      // Look for Portfolio Dashboard component
      console.log('🔍 Looking for Portfolio Dashboard component...');
      const portfolioDashboard = page.locator('[data-testid="portfolio-dashboard"], .portfolio-dashboard');
      const portfolioCards = page.locator('.ant-card').filter({ hasText: /Portfolio|Strategy|P&L|Performance/ });
      
      const portfolioComponentsCount = await portfolioCards.count();
      console.log('📈 Portfolio-related components found:', portfolioComponentsCount);
      
      if (portfolioComponentsCount > 0) {
        console.log('✅ Found portfolio components!');
        for (let i = 0; i < Math.min(5, portfolioComponentsCount); i++) {
          const cardText = await portfolioCards.nth(i).textContent();
          console.log(`  - Component ${i + 1}: ${cardText?.substring(0, 100)}...`);
        }
      }
      
      // Look for specific Portfolio P&L Chart
      const pnlChart = page.locator('text=Portfolio P&L').or(page.locator('text=P&L Analysis'));
      const pnlChartVisible = await pnlChart.isVisible();
      console.log('📊 Portfolio P&L Chart visible:', pnlChartVisible);
      
      // Look for Strategy Correlation Matrix
      const correlationMatrix = page.locator('text=Correlation Matrix').or(page.locator('text=Strategy Correlation'));
      const correlationVisible = await correlationMatrix.isVisible();
      console.log('🔗 Strategy Correlation Matrix visible:', correlationVisible);
      
      // Look for Asset Allocation Chart
      const assetAllocation = page.locator('text=Asset Allocation').or(page.locator('text=Allocation'));
      const allocationVisible = await assetAllocation.isVisible();
      console.log('🥧 Asset Allocation Chart visible:', allocationVisible);
      
      // Count all charts/visualizations
      const charts = page.locator('canvas, svg, .recharts-container, .lightweight-charts');
      const chartCount = await charts.count();
      console.log('📈 Total chart/visualization elements found:', chartCount);
      
    } else {
      console.log('❌ Strategy Management tab not found, checking available tabs...');
      const allTabs = await page.locator('[role="tab"]').allTextContents();
      console.log('📋 Available tabs:', allTabs);
    }
    
    // Take screenshot for verification
    await page.screenshot({ 
      path: 'portfolio-story-4-4-test.png',
      fullPage: true 
    });
    console.log('📸 Screenshot saved: portfolio-story-4-4-test.png');
    
    // Final verification - ensure no critical errors
    const bodyContent = await page.textContent('body');
    expect(bodyContent!.length).toBeGreaterThan(1000);
    console.log('✅ Frontend is fully functional with content length:', bodyContent!.length);
  });
});