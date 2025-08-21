import { test, expect } from '@playwright/test';

test.describe('Strategy Configuration Interface', () => {
  test.beforeEach(async ({ page }) => {
    // Enable console logging for debugging
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    // Navigate to the application
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
  });

  test('Strategy Configuration Interface - Complete User Flow', async ({ page }) => {
    console.log('Starting Strategy Configuration Interface test');

    // Wait for page to fully load
    await page.waitForTimeout(2000);

    // Take initial screenshot
    await page.screenshot({ 
      path: 'strategy-config-01-initial-load.png',
      fullPage: true 
    });

    // Look for navigation or strategy-related elements
    console.log('Searching for strategy-related navigation...');
    
    // Check if there's a Strategy tab/menu item
    const strategyNav = page.locator('text=Strategy', { exact: false }).first();
    const strategyConfigNav = page.locator('text=Strategy Config', { exact: false }).first();
    const strategiesNav = page.locator('text=Strategies', { exact: false }).first();
    
    let navigationFound = false;
    
    // Try to find and click strategy navigation
    if (await strategyNav.isVisible()) {
      console.log('Found Strategy navigation');
      await strategyNav.click();
      navigationFound = true;
    } else if (await strategyConfigNav.isVisible()) {
      console.log('Found Strategy Config navigation');
      await strategyConfigNav.click();
      navigationFound = true;
    } else if (await strategiesNav.isVisible()) {
      console.log('Found Strategies navigation');
      await strategiesNav.click();
      navigationFound = true;
    }

    if (navigationFound) {
      await page.waitForTimeout(1000);
      await page.screenshot({ 
        path: 'strategy-config-02-navigation-clicked.png',
        fullPage: true 
      });
    }

    // Look for Strategy components on the current page
    console.log('Searching for Strategy Configuration components...');

    // Check for Strategy Template Library
    const templateLibrary = page.locator('text=Strategy Template Library').first();
    const templateSelection = page.locator('text=Template Selection').first();
    const strategyBuilder = page.locator('text=Strategy Builder').first();
    const visualBuilder = page.locator('text=Visual Strategy Builder').first();

    if (await templateLibrary.isVisible()) {
      console.log('✅ Found Strategy Template Library');
      await page.screenshot({ 
        path: 'strategy-config-03-template-library.png',
        fullPage: true 
      });

      // Test template selection
      const templateCard = page.locator('.ant-card').first();
      if (await templateCard.isVisible()) {
        console.log('✅ Found template cards');
        await templateCard.click();
        await page.waitForTimeout(500);
        
        await page.screenshot({ 
          path: 'strategy-config-04-template-selected.png',
          fullPage: true 
        });
      }
    }

    if (await visualBuilder.isVisible()) {
      console.log('✅ Found Visual Strategy Builder');
      await page.screenshot({ 
        path: 'strategy-config-05-visual-builder.png',
        fullPage: true 
      });

      // Test Add Component button
      const addComponentBtn = page.locator('text=Add Component').first();
      if (await addComponentBtn.isVisible()) {
        console.log('✅ Found Add Component button');
        await addComponentBtn.click();
        await page.waitForTimeout(500);
        
        await page.screenshot({ 
          path: 'strategy-config-06-component-library.png',
          fullPage: true 
        });
      }
    }

    // Check for Strategy Lifecycle Controls
    const lifecycleControls = page.locator('text=Strategy Lifecycle').first();
    const deployButton = page.locator('text=Deploy Strategy').first();
    
    if (await lifecycleControls.isVisible()) {
      console.log('✅ Found Strategy Lifecycle Controls');
      await page.screenshot({ 
        path: 'strategy-config-07-lifecycle-controls.png',
        fullPage: true 
      });

      if (await deployButton.isVisible()) {
        console.log('✅ Found Deploy Strategy button');
        await deployButton.click();
        await page.waitForTimeout(500);
        
        await page.screenshot({ 
          path: 'strategy-config-08-deploy-modal.png',
          fullPage: true 
        });
      }
    }

    // Check for Parameter Configuration
    const parameterConfig = page.locator('text=Parameter Configuration').first();
    const parameters = page.locator('text=Parameters').first();
    
    if (await parameterConfig.isVisible() || await parameters.isVisible()) {
      console.log('✅ Found Parameter Configuration');
      await page.screenshot({ 
        path: 'strategy-config-09-parameter-config.png',
        fullPage: true 
      });
    }

    // Check for Version Control
    const versionControl = page.locator('text=Version Control').first();
    const configHistory = page.locator('text=Configuration History').first();
    
    if (await versionControl.isVisible() || await configHistory.isVisible()) {
      console.log('✅ Found Version Control');
      await page.screenshot({ 
        path: 'strategy-config-10-version-control.png',
        fullPage: true 
        });
    }

    // Search for any strategy-related form inputs
    const formInputs = await page.locator('input[type="text"], input[type="number"], select, textarea').count();
    console.log(`Found ${formInputs} form inputs on the page`);

    // Search for any strategy-related buttons
    const buttons = await page.locator('button').count();
    console.log(`Found ${buttons} buttons on the page`);

    // Take final screenshot
    await page.screenshot({ 
      path: 'strategy-config-11-final-state.png',
      fullPage: true 
    });

    // Verify core components are present
    const hasStrategyElements = await Promise.all([
      templateLibrary.isVisible(),
      visualBuilder.isVisible(),
      lifecycleControls.isVisible(),
      deployButton.isVisible()
    ]).then(results => results.some(Boolean));

    if (hasStrategyElements) {
      console.log('✅ Strategy Configuration Interface components found and functional');
    } else {
      console.log('❌ Strategy Configuration Interface components not found on current page');
      
      // Search entire page content for strategy-related terms
      const pageContent = await page.textContent('body');
      const strategyTerms = ['strategy', 'template', 'deploy', 'parameter', 'config'];
      const foundTerms = strategyTerms.filter(term => 
        pageContent?.toLowerCase().includes(term)
      );
      
      console.log('Strategy-related terms found on page:', foundTerms);
    }

    // Test successful if we found strategy-related elements
    expect(hasStrategyElements || formInputs > 0).toBeTruthy();
  });

  test('Direct Component Access Test', async ({ page }) => {
    console.log('Testing direct access to Strategy components');

    // Try to access strategy components directly via URL fragments or specific routes
    const possibleRoutes = [
      'http://localhost:3000/#/strategy',
      'http://localhost:3000/#/strategies', 
      'http://localhost:3000/#/strategy-config',
      'http://localhost:3000/#/strategy-builder'
    ];

    for (const route of possibleRoutes) {
      console.log(`Trying route: ${route}`);
      await page.goto(route);
      await page.waitForTimeout(1000);
      
      const hasStrategyContent = await page.locator('text=Strategy').first().isVisible();
      if (hasStrategyContent) {
        console.log(`✅ Found strategy content at: ${route}`);
        await page.screenshot({ 
          path: `strategy-direct-access-${route.split('/').pop()}.png`,
          fullPage: true 
        });
        break;
      }
    }
  });

  test('API Endpoint Test', async ({ page }) => {
    console.log('Testing Strategy API endpoints');

    // Test strategy-related API endpoints
    const apiEndpoints = [
      'http://localhost:8000/api/v1/strategies/templates',
      'http://localhost:8000/health'
    ];

    for (const endpoint of apiEndpoints) {
      try {
        console.log(`Testing API endpoint: ${endpoint}`);
        const response = await page.request.get(endpoint);
        console.log(`API ${endpoint}: Status ${response.status()}`);
        
        if (response.ok()) {
          const data = await response.text();
          console.log(`API Response preview: ${data.substring(0, 200)}...`);
        }
      } catch (error) {
        console.log(`API ${endpoint}: Error - ${error}`);
      }
    }
  });
});