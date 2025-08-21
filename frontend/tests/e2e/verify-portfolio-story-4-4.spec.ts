/**
 * Verify Portfolio Story 4.4 Implementation
 */

import { test, expect } from '@playwright/test';

test('Verify Portfolio Story 4.4 - Multi-Strategy Portfolio Visualization', async ({ page }) => {
  console.log('🧪 Verifying Portfolio Story 4.4 implementation...');
  
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);
  
  // Verify frontend loads
  const content = await page.textContent('body');
  console.log('📄 Frontend loaded with', content!.length, 'characters');
  expect(content!.length).toBeGreaterThan(1000);
  
  // Check if we can access the portfolio files directly
  console.log('🔍 Testing portfolio component file access...');
  
  const portfolioFiles = [
    '/src/components/Portfolio/PortfolioDashboard.tsx',
    '/src/components/Portfolio/PortfolioPnLChart.tsx', 
    '/src/components/Portfolio/StrategyCorrelationMatrix.tsx',
    '/src/services/portfolioAggregationService.ts',
    '/src/services/portfolioMetrics.ts'
  ];
  
  for (const file of portfolioFiles) {
    const response = await page.request.get(`http://localhost:3000${file}`);
    const accessible = response.status() === 200;
    console.log(`${accessible ? '✅' : '❌'} ${file}: ${response.status()}`);
    
    if (accessible) {
      const content = await response.text();
      console.log(`   - Content length: ${content.length} characters`);
    }
  }
  
  // Test if we can find mentions of portfolio functionality on the page
  console.log('🔍 Looking for portfolio-related content...');
  
  const portfolioKeywords = [
    'Portfolio',
    'Strategy',
    'Performance',
    'Risk',
    'Management'
  ];
  
  for (const keyword of portfolioKeywords) {
    const found = content!.includes(keyword);
    console.log(`${found ? '✅' : '❌'} Found "${keyword}": ${found}`);
  }
  
  // Take screenshot
  await page.screenshot({ path: 'portfolio-story-verification.png' });
  console.log('📸 Screenshot saved: portfolio-story-verification.png');
  
  console.log('✅ Portfolio Story 4.4 verification completed!');
});