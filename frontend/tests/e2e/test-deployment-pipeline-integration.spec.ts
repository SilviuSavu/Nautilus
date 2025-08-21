import { test, expect } from '@playwright/test';

test.describe('Strategy Deployment Pipeline Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the strategy management page
    await page.goto('http://localhost:3000');
    
    // Wait for the application to load
    await page.waitForSelector('[data-testid="app-loaded"]', { timeout: 10000 });
    
    // Navigate to strategy deployment
    await page.click('[data-testid="strategy-tab"]');
    await page.waitForSelector('[data-testid="strategy-deployment-section"]');
  });

  test('Complete deployment workflow - Create to Approval', async ({ page }) => {
    // Step 1: Create new deployment request
    await page.click('[data-testid="new-deployment-button"]');
    await page.waitForSelector('[data-testid="deployment-pipeline-modal"]');
    
    // Verify initial step
    await expect(page.locator('[data-testid="deployment-step-configuration"]')).toBeVisible();
    await expect(page.locator('text=Configuration')).toHaveClass(/ant-steps-item-active/);
    
    // Fill deployment configuration
    await page.fill('[data-testid="strategy-version-input"]', '2.1.0');
    await page.selectOption('[data-testid="environment-select"]', 'production');
    await page.fill('[data-testid="initial-position-size"]', '25');
    await page.fill('[data-testid="max-daily-loss"]', '1000');
    await page.fill('[data-testid="max-positions"]', '5');
    await page.check('[data-testid="enable-gradual-rollout"]');
    
    // Configure approval settings
    await page.check('[data-testid="approval-required"]');
    await page.selectOption('[data-testid="required-approvals"]', ['senior_trader', 'risk_manager']);
    
    // Advance to risk assessment
    await page.click('[data-testid="next-button"]');
    await page.waitForSelector('[data-testid="deployment-step-risk-assessment"]');
    
    // Step 2: Risk Assessment
    await expect(page.locator('text=Risk Assessment')).toHaveClass(/ant-steps-item-active/);
    
    // Trigger risk assessment
    await page.click('[data-testid="perform-risk-assessment-button"]');
    await page.waitForSelector('[data-testid="risk-assessment-results"]', { timeout: 10000 });
    
    // Verify risk assessment results
    await expect(page.locator('[data-testid="risk-level-badge"]')).toContainText(/LOW|MEDIUM|HIGH/);
    await expect(page.locator('[data-testid="portfolio-impact"]')).toBeVisible();
    await expect(page.locator('[data-testid="max-drawdown-estimate"]')).toBeVisible();
    await expect(page.locator('[data-testid="var-estimate"]')).toBeVisible();
    
    // Check for warnings or blockers
    const warnings = await page.locator('[data-testid="risk-warnings"]');
    if (await warnings.isVisible()) {
      console.log('Risk warnings detected');
    }
    
    const blockers = await page.locator('[data-testid="risk-blockers"]');
    if (await blockers.isVisible()) {
      await expect(page.locator('[data-testid="next-button"]')).toBeDisabled();
      console.log('Risk blockers detected - cannot proceed');
      return;
    }
    
    // Advance to review step
    await page.click('[data-testid="next-button"]');
    await page.waitForSelector('[data-testid="deployment-step-review"]');
    
    // Step 3: Review Configuration
    await expect(page.locator('text=Review')).toHaveClass(/ant-steps-item-active/);
    
    // Verify configuration tab
    await expect(page.locator('[data-testid="config-strategy-name"]')).toBeVisible();
    await expect(page.locator('[data-testid="config-version"]')).toContainText('2.1.0');
    await expect(page.locator('[data-testid="config-environment"]')).toContainText('production');
    
    // Check backtest results tab
    await page.click('[data-testid="backtest-results-tab"]');
    await page.waitForSelector('[data-testid="backtest-total-return"]');
    await expect(page.locator('[data-testid="backtest-sharpe-ratio"]')).toBeVisible();
    await expect(page.locator('[data-testid="backtest-max-drawdown"]')).toBeVisible();
    
    // Check rollout plan tab
    await page.click('[data-testid="rollout-plan-tab"]');
    await page.waitForSelector('[data-testid="rollout-phases-table"]');
    
    const phaseRows = await page.locator('[data-testid="rollout-phase-row"]').count();
    expect(phaseRows).toBeGreaterThan(0);
    
    // Advance to approval step
    await page.click('[data-testid="next-button"]');
    await page.waitForSelector('[data-testid="deployment-step-approval"]');
    
    // Step 4: Submit for Approval
    await expect(page.locator('text=Approval')).toHaveClass(/ant-steps-item-active/);
    
    // Submit deployment request
    await page.click('[data-testid="submit-approval-button"]');
    
    // Wait for submission success
    await page.waitForSelector('[data-testid="deployment-submitted-alert"]', { timeout: 10000 });
    await expect(page.locator('[data-testid="deployment-id"]')).toBeVisible();
    
    // Verify approval status
    await expect(page.locator('[data-testid="deployment-status"]')).toContainText('PENDING_APPROVAL');
    await expect(page.locator('[data-testid="approval-progress"]')).toBeVisible();
    
    // Check required approvals list
    const pendingApprovals = await page.locator('[data-testid="pending-approvals"]');
    await expect(pendingApprovals).toContainText('senior_trader');
    await expect(pendingApprovals).toContainText('risk_manager');
  });

  test('Live Strategy Monitoring Interface', async ({ page }) => {
    // Navigate to live strategies monitoring
    await page.click('[data-testid="live-strategies-tab"]');
    await page.waitForSelector('[data-testid="live-strategies-monitor"]');
    
    // Create a mock live strategy for testing
    await page.evaluate(() => {
      // Mock live strategy data
      window.mockLiveStrategy = {
        strategyInstanceId: 'test-strategy-instance-123',
        strategyId: 'momentum-breakout-v2.1.0',
        version: '2.1.0',
        state: 'running',
        realizedPnL: 1250.75,
        unrealizedPnL: -45.30,
        currentPosition: {
          instrument: 'EURUSD',
          side: 'LONG',
          quantity: 100000,
          avgPrice: 1.1050,
          marketValue: 110500
        },
        performanceMetrics: {
          totalPnL: 1205.45,
          totalTrades: 15,
          winRate: 0.67,
          dailyPnL: 125.30
        },
        riskMetrics: {
          currentDrawdown: 2.3,
          valueAtRisk: 450.0,
          leverageRatio: 1.5
        },
        healthStatus: {
          overall: 'healthy',
          heartbeat: 'active',
          dataFeed: 'connected',
          orderExecution: 'normal',
          riskCompliance: 'compliant'
        }
      };
    });
    
    // Verify strategy overview cards
    await expect(page.locator('[data-testid="strategy-total-pnl"]')).toBeVisible();
    await expect(page.locator('[data-testid="strategy-daily-pnl"]')).toBeVisible();
    await expect(page.locator('[data-testid="strategy-drawdown"]')).toBeVisible();
    await expect(page.locator('[data-testid="strategy-trades"]')).toBeVisible();
    
    // Check position information
    await expect(page.locator('[data-testid="current-position-instrument"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-position-side"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-position-quantity"]')).toBeVisible();
    
    // Verify risk metrics
    await expect(page.locator('[data-testid="risk-var"]')).toBeVisible();
    await expect(page.locator('[data-testid="risk-leverage"]')).toBeVisible();
    await expect(page.locator('[data-testid="risk-correlation"]')).toBeVisible();
    
    // Check health status indicators
    await expect(page.locator('[data-testid="health-heartbeat"]')).toBeVisible();
    await expect(page.locator('[data-testid="health-data-feed"]')).toBeVisible();
    await expect(page.locator('[data-testid="health-order-execution"]')).toBeVisible();
    await expect(page.locator('[data-testid="health-risk-compliance"]')).toBeVisible();
    
    // Test strategy controls
    const pauseButton = page.locator('[data-testid="pause-strategy-button"]');
    if (await pauseButton.isVisible()) {
      await pauseButton.click();
      await page.waitForSelector('[data-testid="strategy-paused-notification"]');
      await expect(page.locator('[data-testid="strategy-state"]')).toContainText('paused');
      
      // Test resume
      await page.click('[data-testid="resume-strategy-button"]');
      await page.waitForSelector('[data-testid="strategy-resumed-notification"]');
      await expect(page.locator('[data-testid="strategy-state"]')).toContainText('running');
    }
  });

  test('Emergency Control Panel Functionality', async ({ page }) => {
    // Navigate to emergency controls
    await page.click('[data-testid="emergency-controls-tab"]');
    await page.waitForSelector('[data-testid="emergency-control-panel"]');
    
    // Verify system overview
    await expect(page.locator('[data-testid="total-strategies"]')).toBeVisible();
    await expect(page.locator('[data-testid="running-strategies"]')).toBeVisible();
    await expect(page.locator('[data-testid="paused-strategies"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-strategies"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-exposure"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-pnl"]')).toBeVisible();
    
    // Check strategy table
    await expect(page.locator('[data-testid="strategies-table"]')).toBeVisible();
    
    // Test emergency stop confirmation
    const emergencyButton = page.locator('[data-testid="emergency-stop-strategy-button"]').first();
    if (await emergencyButton.isVisible()) {
      await emergencyButton.click();
      
      // Verify confirmation modal
      await page.waitForSelector('[data-testid="emergency-stop-confirmation"]');
      await expect(page.locator('text=EMERGENCY STOP CONFIRMATION')).toBeVisible();
      await expect(page.locator('[data-testid="emergency-confirmation-input"]')).toBeVisible();
      
      // Test cancellation
      await page.click('[data-testid="cancel-emergency-stop"]');
      await expect(page.locator('[data-testid="emergency-stop-confirmation"]')).not.toBeVisible();
    }
    
    // Test bulk actions
    await page.check('[data-testid="strategy-checkbox"]', { first: true });
    await expect(page.locator('[data-testid="bulk-actions-button"]')).toBeEnabled();
    
    await page.click('[data-testid="bulk-actions-button"]');
    await page.waitForSelector('[data-testid="bulk-actions-modal"]');
    
    await page.selectOption('[data-testid="bulk-action-select"]', 'pause');
    await page.fill('[data-testid="bulk-action-reason"]', 'Maintenance pause');
    
    await page.click('[data-testid="execute-bulk-action"]');
    await page.waitForSelector('[data-testid="bulk-action-success"]');
  });

  test('Gradual Rollout Progress Monitoring', async ({ page }) => {
    // Navigate to rollout controls
    await page.click('[data-testid="rollout-controls-tab"]');
    await page.waitForSelector('[data-testid="gradual-rollout-controls"]');
    
    // Verify phase steps display
    await expect(page.locator('[data-testid="rollout-phase-steps"]')).toBeVisible();
    
    const phases = await page.locator('[data-testid="rollout-phase"]').count();
    expect(phases).toBeGreaterThan(0);
    
    // Check current phase status
    await expect(page.locator('[data-testid="current-phase-name"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-phase-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="phase-duration"]')).toBeVisible();
    
    // Verify success criteria display
    await expect(page.locator('[data-testid="success-criteria"]')).toBeVisible();
    
    // Test phase advancement (if criteria are met)
    const advanceButton = page.locator('[data-testid="advance-phase-button"]');
    if (await advanceButton.isVisible() && await advanceButton.isEnabled()) {
      await advanceButton.click();
      await page.waitForSelector('[data-testid="phase-advanced-notification"]');
    }
    
    // Test rollout modification
    await page.click('[data-testid="modify-rollout-button"]');
    await page.waitForSelector('[data-testid="modify-rollout-modal"]');
    
    // Verify phase editing form
    await expect(page.locator('[data-testid="phase-editor"]')).toBeVisible();
    
    // Test adding a phase
    await page.click('[data-testid="add-phase-button"]');
    await page.fill('[data-testid="new-phase-name"]', 'custom_phase');
    await page.fill('[data-testid="new-phase-position-size"]', '75');
    await page.fill('[data-testid="new-phase-duration"]', '3600');
    
    // Cancel modification
    await page.click('[data-testid="cancel-modification"]');
    await expect(page.locator('[data-testid="modify-rollout-modal"]')).not.toBeVisible();
  });

  test('Strategy Health Dashboard', async ({ page }) => {
    // Navigate to health dashboard
    await page.click('[data-testid="health-dashboard-tab"]');
    await page.waitForSelector('[data-testid="strategy-health-dashboard"]');
    
    // Verify system overview
    await expect(page.locator('[data-testid="system-health-badge"]')).toBeVisible();
    await expect(page.locator('[data-testid="active-strategies-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-alerts-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="critical-alerts-count"]')).toBeVisible();
    
    // Check data connections status
    await expect(page.locator('[data-testid="data-connections"]')).toBeVisible();
    
    const connections = await page.locator('[data-testid="connection-status"]').count();
    expect(connections).toBeGreaterThan(0);
    
    // Verify strategy health table
    await expect(page.locator('[data-testid="strategy-health-table"]')).toBeVisible();
    
    // Check alerts section
    await expect(page.locator('[data-testid="system-alerts"]')).toBeVisible();
    
    // Test alert filtering
    await page.selectOption('[data-testid="alert-filter-select"]', 'critical');
    await page.waitForSelector('[data-testid="filtered-alerts"]');
    
    // Test strategy filtering
    await page.selectOption('[data-testid="strategy-filter-select"]', 'all');
    
    // Test alert acknowledgment
    const acknowledgeButton = page.locator('[data-testid="acknowledge-alert-button"]').first();
    if (await acknowledgeButton.isVisible()) {
      await acknowledgeButton.click();
      await page.waitForSelector('[data-testid="alert-acknowledged-notification"]');
    }
    
    // Test refresh functionality
    await page.click('[data-testid="refresh-health-data"]');
    await page.waitForSelector('[data-testid="health-data-refreshed"]');
  });

  test('API Integration and Error Handling', async ({ page }) => {
    // Test API error scenarios
    
    // Mock API failures
    await page.route('**/api/v1/nautilus/deployment/create', route => {
      route.fulfill({ status: 500, body: JSON.stringify({ error: 'Internal server error' }) });
    });
    
    // Attempt to create deployment with mocked failure
    await page.click('[data-testid="new-deployment-button"]');
    await page.waitForSelector('[data-testid="deployment-pipeline-modal"]');
    
    // Fill minimal required fields
    await page.fill('[data-testid="strategy-version-input"]', '2.1.0');
    
    // Try to submit
    await page.click('[data-testid="next-button"]');
    await page.click('[data-testid="perform-risk-assessment-button"]');
    
    // Navigate to approval step and submit
    await page.click('[data-testid="next-button"]');
    await page.click('[data-testid="next-button"]');
    await page.click('[data-testid="submit-approval-button"]');
    
    // Verify error handling
    await page.waitForSelector('[data-testid="api-error-notification"]');
    await expect(page.locator('text=Failed to create deployment request')).toBeVisible();
    
    // Test retry functionality
    await page.unroute('**/api/v1/nautilus/deployment/create');
    await page.route('**/api/v1/nautilus/deployment/create', route => {
      route.fulfill({ 
        status: 200, 
        body: JSON.stringify({ 
          deploymentId: 'retry-deployment-123',
          status: 'pending_approval'
        })
      });
    });
    
    await page.click('[data-testid="retry-submission-button"]');
    await page.waitForSelector('[data-testid="deployment-submitted-alert"]');
  });

  test('Real-time Data Updates', async ({ page }) => {
    // Navigate to live monitoring
    await page.click('[data-testid="live-strategies-tab"]');
    await page.waitForSelector('[data-testid="live-strategies-monitor"]');
    
    // Mock WebSocket or polling updates
    await page.evaluate(() => {
      let updateCount = 0;
      setInterval(() => {
        updateCount++;
        
        // Simulate metric updates
        const event = new CustomEvent('strategy-metrics-update', {
          detail: {
            strategyInstanceId: 'test-strategy-instance-123',
            performanceMetrics: {
              dailyPnL: 125.30 + (updateCount * 10),
              totalTrades: 15 + updateCount
            },
            riskMetrics: {
              currentDrawdown: 2.3 + (Math.random() * 0.5)
            },
            timestamp: new Date()
          }
        });
        
        window.dispatchEvent(event);
      }, 2000);
    });
    
    // Verify initial values
    const initialPnL = await page.locator('[data-testid="strategy-daily-pnl"]').textContent();
    
    // Wait for updates
    await page.waitForTimeout(5000);
    
    // Verify values have changed
    const updatedPnL = await page.locator('[data-testid="strategy-daily-pnl"]').textContent();
    expect(updatedPnL).not.toBe(initialPnL);
    
    // Check last update timestamp
    await expect(page.locator('[data-testid="last-update-time"]')).toBeVisible();
  });

  test('Performance and Responsiveness', async ({ page }) => {
    // Test application performance with large datasets
    
    // Navigate to strategies overview
    await page.goto('http://localhost:3000/strategies');
    
    // Measure page load time
    const startTime = Date.now();
    await page.waitForSelector('[data-testid="strategies-loaded"]', { timeout: 30000 });
    const loadTime = Date.now() - startTime;
    
    console.log(`Strategies page load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds
    
    // Test table scrolling performance with many rows
    const tableContainer = page.locator('[data-testid="strategies-table-container"]');
    if (await tableContainer.isVisible()) {
      // Scroll through the table
      await tableContainer.evaluate(el => {
        el.scrollTop = el.scrollHeight;
      });
      
      // Verify scrolling is smooth (no major lag)
      await page.waitForTimeout(1000);
      await expect(tableContainer).toBeVisible();
    }
    
    // Test filter responsiveness
    const filterInput = page.locator('[data-testid="strategy-filter-input"]');
    if (await filterInput.isVisible()) {
      const filterStart = Date.now();
      await filterInput.fill('momentum');
      await page.waitForSelector('[data-testid="filtered-results"]', { timeout: 3000 });
      const filterTime = Date.now() - filterStart;
      
      console.log(`Filter response time: ${filterTime}ms`);
      expect(filterTime).toBeLessThan(1000); // Should filter within 1 second
    }
  });
});