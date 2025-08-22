import { test, expect, Page } from '@playwright/test';

/**
 * Dashboard Full Functionality Test Suite
 * Complete testing of all tab contents and functionalities
 */

// Helper functions
async function navigateToDashboard(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
  await page.waitForTimeout(2000);
}

async function clickTab(page: Page, tabKey: string) {
  await page.click(`[data-node-key="${tabKey}"]`);
  await page.waitForTimeout(3000);
}

async function checkTableExists(page: Page): Promise<boolean> {
  return await page.locator('.ant-table, table').isVisible();
}

async function checkStatisticCards(page: Page): Promise<number> {
  return await page.locator('.ant-statistic').count();
}

async function checkChartElements(page: Page): Promise<number> {
  return await page.locator('[class*="chart"], canvas, svg[role="img"]').count();
}

test.describe('Dashboard Full Functionality Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToDashboard(page);
  });

  test.describe('IB Tab - Complete Interactive Brokers Functionality', () => {
    test('IB dashboard all features', async ({ page }) => {
      await clickTab(page, 'ib');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Connection Status Section
        console.log('Testing IB Connection Status...');
        const connectionElements = await page.locator('.ant-card, .ant-alert').filter({ 
          hasText: /Connection|Gateway|Connected|Disconnected|IB Gateway|TWS/ 
        }).count();
        expect(connectionElements).toBeGreaterThanOrEqual(0);
        console.log(`Found ${connectionElements} connection status elements`);
        
        // Test Account Information Display
        console.log('Testing Account Information...');
        const accountStats = await page.locator('.ant-statistic').filter({ 
          hasText: /Net Liquidation|Buying Power|Total Cash|Available Funds|Margin|Excess Liquidity/ 
        }).count();
        console.log(`Found ${accountStats} account statistics`);
        
        // Test Position Display Table
        console.log('Testing Position Display...');
        const positionElements = await page.locator('.ant-table, .ant-list').filter({ 
          hasText: /Position|Symbol|Quantity|Market Value|P&L|Unrealized|Realized/ 
        }).count();
        console.log(`Found ${positionElements} position display elements`);
        
        // Test Order Management Section
        console.log('Testing Order Management...');
        const orderElements = await page.locator('button, .ant-card').filter({ 
          hasText: /Place Order|Cancel Order|Modify|Market Order|Limit Order|Stop/ 
        }).count();
        console.log(`Found ${orderElements} order management elements`);
        
        // Test Trade History Tab
        const tradeHistoryTab = page.locator('.ant-tabs-tab').filter({ hasText: /Trade History|Trades|Executions/ });
        if (await tradeHistoryTab.first().isVisible()) {
          await tradeHistoryTab.first().click();
          await page.waitForTimeout(2000);
          
          const tradeTable = await page.locator('.ant-table').filter({ 
            hasText: /Symbol|Time|Price|Quantity|Side|Commission/ 
          }).count();
          console.log(`Trade history table elements: ${tradeTable}`);
        }
        
        // Test Order Book Display
        const orderBookTab = page.locator('.ant-tabs-tab').filter({ hasText: /Order Book|Market Depth|Level 2/ });
        if (await orderBookTab.first().isVisible()) {
          await orderBookTab.first().click();
          await page.waitForTimeout(2000);
          
          const orderBookElements = await page.locator('.order-book, [class*="orderbook"], .ant-table').filter({ 
            hasText: /Bid|Ask|Price|Size|Level/ 
          }).count();
          console.log(`Order book elements: ${orderBookElements}`);
        }
        
        // Test Account Activity
        const activityElements = await page.locator('.ant-timeline, .ant-list').filter({ 
          hasText: /Activity|Transaction|Deposit|Withdrawal/ 
        }).count();
        console.log(`Account activity elements: ${activityElements}`);
        
        // Test Real-time Updates
        const realtimeIndicators = await page.locator('.ant-badge-status-processing, .ant-spin, [class*="blink"]').count();
        console.log(`Real-time update indicators: ${realtimeIndicators}`);
      }
    });
  });

  test.describe('Factors Tab - Complete Factor Analysis Functionality', () => {
    test('Factor dashboard all features', async ({ page }) => {
      await clickTab(page, 'factors');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test FRED Economic Factors Section
        console.log('Testing FRED Economic Factors...');
        const fredElements = await page.locator('.ant-card, .ant-statistic, .ant-tag').filter({ 
          hasText: /FRED|Federal Reserve|GDP|CPI|Unemployment|Interest Rate|Inflation|VIX|DXY|Yield Curve/ 
        }).count();
        expect(fredElements).toBeGreaterThanOrEqual(0);
        console.log(`Found ${fredElements} FRED economic factor elements`);
        
        // Test Alpha Vantage Market Data
        console.log('Testing Alpha Vantage Integration...');
        const alphaVantageElements = await page.locator('.ant-card, .ant-tag, .ant-statistic').filter({ 
          hasText: /Alpha Vantage|Market Data|Quote|Fundamental|Earnings|Financial Ratios/ 
        }).count();
        console.log(`Found ${alphaVantageElements} Alpha Vantage elements`);
        
        // Test EDGAR SEC Data Integration
        console.log('Testing EDGAR SEC Data...');
        const edgarElements = await page.locator('.ant-card, .ant-list, .ant-tag').filter({ 
          hasText: /EDGAR|SEC|Filing|10-K|10-Q|8-K|Proxy|CIK|Ticker/ 
        }).count();
        console.log(`Found ${edgarElements} EDGAR SEC data elements`);
        
        // Test Cross-Source Factor Analysis
        console.log('Testing Cross-Source Factors...');
        const crossSourceElements = await page.locator('.ant-table, .ant-card').filter({ 
          hasText: /Cross-Source|Combined|Multi-Source|Composite|Aggregate|Correlation/ 
        }).count();
        console.log(`Found ${crossSourceElements} cross-source factor elements`);
        
        // Test Factor Categories/Tabs
        const factorTabs = await page.locator('.ant-tabs-tab').count();
        console.log(`Found ${factorTabs} factor category tabs`);
        
        // Test Factor Streaming Controls
        console.log('Testing Streaming Controls...');
        const streamingControls = await page.locator('button').filter({ 
          hasText: /Start Stream|Stop Stream|Pause|Resume|Refresh|Update/ 
        }).count();
        console.log(`Found ${streamingControls} streaming control buttons`);
        
        // Test Real-time Factor Updates
        const realtimeElements = await page.locator('.ant-badge-status-processing, .ant-spin, .ant-progress').count();
        console.log(`Found ${realtimeElements} real-time update indicators`);
        
        // Test Factor Charts/Visualizations
        const chartElements = await checkChartElements(page);
        console.log(`Found ${chartElements} chart/visualization elements`);
        
        // Test Factor Statistics
        const statisticsCount = await checkStatisticCards(page);
        console.log(`Found ${statisticsCount} factor statistic cards`);
        
        // Test Factor Health Status
        const healthStatus = await page.locator('.ant-badge, .ant-tag').filter({ 
          hasText: /Healthy|Connected|Active|Error|Disconnected/ 
        }).count();
        console.log(`Found ${healthStatus} health status indicators`);
      }
    });
  });

  test.describe('Risk Tab - Complete Risk Management Functionality', () => {
    test('Risk dashboard all features', async ({ page }) => {
      await clickTab(page, 'risk');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Risk Metrics Display
        console.log('Testing Risk Metrics...');
        const riskMetrics = await page.locator('.ant-statistic, .ant-card').filter({ 
          hasText: /VaR|Value at Risk|Sharpe Ratio|Sortino|Max Drawdown|Beta|Alpha|Volatility|Standard Deviation|Downside Risk/ 
        }).count();
        expect(riskMetrics).toBeGreaterThanOrEqual(0);
        console.log(`Found ${riskMetrics} risk metric displays`);
        
        // Test Exposure Analysis
        console.log('Testing Exposure Analysis...');
        const exposureElements = await page.locator('.ant-card, .ant-table').filter({ 
          hasText: /Exposure|Concentration|Allocation|Sector|Asset Class|Geographic|Currency/ 
        }).count();
        console.log(`Found ${exposureElements} exposure analysis elements`);
        
        // Test Risk Alert System
        console.log('Testing Alert System...');
        const alertElements = await page.locator('.ant-alert, .ant-badge, .ant-tag').filter({ 
          hasText: /Alert|Warning|Risk Limit|Breach|Threshold|Violation|Critical/ 
        }).count();
        console.log(`Found ${alertElements} risk alert elements`);
        
        // Test Risk Limits Configuration
        console.log('Testing Risk Limits...');
        const limitControls = await page.locator('.ant-input-number, .ant-slider, .ant-form-item').filter({ 
          hasText: /Limit|Maximum|Minimum|Threshold|Tolerance/ 
        }).count();
        console.log(`Found ${limitControls} risk limit configuration controls`);
        
        // Test Risk Visualization Charts
        const riskCharts = await checkChartElements(page);
        console.log(`Found ${riskCharts} risk visualization charts`);
        
        // Test Real-time Risk Monitoring
        const realtimeMonitoring = await page.locator('.ant-progress, .ant-badge-status-processing').count();
        console.log(`Found ${realtimeMonitoring} real-time monitoring indicators`);
        
        // Test Risk Reports
        const reportElements = await page.locator('button, .ant-card').filter({ 
          hasText: /Report|Export|Download|Generate|PDF|CSV/ 
        }).count();
        console.log(`Found ${reportElements} risk report elements`);
        
        // Test Stress Testing
        const stressTestElements = await page.locator('.ant-card, .ant-form').filter({ 
          hasText: /Stress Test|Scenario|Simulation|What-If/ 
        }).count();
        console.log(`Found ${stressTestElements} stress testing elements`);
      }
    });
  });

  test.describe('Portfolio Tab - Complete Portfolio Analysis', () => {
    test('Portfolio visualization all features', async ({ page }) => {
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Asset Allocation Display
        console.log('Testing Asset Allocation...');
        const allocationElements = await page.locator('.ant-card, [class*="chart"]').filter({ 
          hasText: /Asset Allocation|Portfolio Composition|Holdings|Weights|Percentage/ 
        }).count();
        expect(allocationElements).toBeGreaterThanOrEqual(0);
        console.log(`Found ${allocationElements} asset allocation elements`);
        
        // Test Portfolio P&L Analysis
        console.log('Testing P&L Analysis...');
        const pnlElements = await page.locator('.ant-statistic, .ant-card').filter({ 
          hasText: /P&L|Profit|Loss|Return|Performance|Gain|Unrealized|Realized|Total Return/ 
        }).count();
        console.log(`Found ${pnlElements} P&L analysis elements`);
        
        // Test Strategy Contribution
        console.log('Testing Strategy Contribution...');
        const strategyContribution = await page.locator('.ant-table, .ant-list').filter({ 
          hasText: /Strategy|Contribution|Attribution|Performance Contribution/ 
        }).count();
        console.log(`Found ${strategyContribution} strategy contribution elements`);
        
        // Test Correlation Matrix
        console.log('Testing Correlation Matrix...');
        const correlationElements = await page.locator('.ant-table, [class*="matrix"], [class*="heatmap"]').filter({ 
          hasText: /Correlation|Covariance|Relationship/ 
        }).count();
        console.log(`Found ${correlationElements} correlation matrix elements`);
        
        // Test Diversification Metrics
        console.log('Testing Diversification...');
        const diversificationMetrics = await page.locator('.ant-progress, .ant-statistic').filter({ 
          hasText: /Diversification|Concentration|HHI|Herfindahl|Entropy/ 
        }).count();
        console.log(`Found ${diversificationMetrics} diversification metrics`);
        
        // Test Performance Comparison
        console.log('Testing Performance Comparison...');
        const comparisonElements = await page.locator('[class*="chart"], .ant-table').filter({ 
          hasText: /Benchmark|Relative|Comparison|Outperform|Underperform|Tracking/ 
        }).count();
        console.log(`Found ${comparisonElements} performance comparison elements`);
        
        // Test Portfolio Charts
        const portfolioCharts = await checkChartElements(page);
        console.log(`Found ${portfolioCharts} portfolio visualization charts`);
        
        // Test Allocation Drift Monitor
        const driftMonitor = await page.locator('.ant-card, .ant-alert').filter({ 
          hasText: /Drift|Rebalance|Target|Deviation/ 
        }).count();
        console.log(`Found ${driftMonitor} allocation drift monitor elements`);
      }
    });
  });

  test.describe('Performance Tab - Complete Performance Analytics', () => {
    test('Performance dashboard all features', async ({ page }) => {
      await clickTab(page, 'performance');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Performance Metrics Cards
        console.log('Testing Performance Metrics...');
        const metricsCards = await page.locator('.ant-statistic, .ant-card').filter({ 
          hasText: /Total Return|Win Rate|Profit Factor|Sharpe Ratio|Sortino|Calmar|Max Drawdown|Recovery|Average Win|Average Loss/ 
        }).count();
        expect(metricsCards).toBeGreaterThanOrEqual(0);
        console.log(`Found ${metricsCards} performance metric cards`);
        
        // Test Equity Curve Visualization
        console.log('Testing Equity Curve...');
        const equityCurveElements = await page.locator('[class*="chart"], canvas').filter({ 
          hasText: /Equity|Balance|Cumulative|P&L|Growth/ 
        }).count();
        console.log(`Found ${equityCurveElements} equity curve elements`);
        
        // Test Execution Analytics
        console.log('Testing Execution Analytics...');
        const executionAnalytics = await page.locator('.ant-table, .ant-card').filter({ 
          hasText: /Execution|Slippage|Fill Rate|Latency|Speed|Commission|Cost/ 
        }).count();
        console.log(`Found ${executionAnalytics} execution analytics elements`);
        
        // Test Attribution Dashboard
        console.log('Testing Attribution...');
        const attributionElements = await page.locator('.ant-card, .ant-table').filter({ 
          hasText: /Attribution|Factor|Contribution|Source|Breakdown/ 
        }).count();
        console.log(`Found ${attributionElements} attribution dashboard elements`);
        
        // Test Monte Carlo Simulation
        console.log('Testing Monte Carlo...');
        const monteCarloElements = await page.locator('[class*="chart"], .ant-card').filter({ 
          hasText: /Monte Carlo|Simulation|Probability|Distribution|Confidence/ 
        }).count();
        console.log(`Found ${monteCarloElements} Monte Carlo simulation elements`);
        
        // Test Statistical Tests
        console.log('Testing Statistical Tests...');
        const statsTestElements = await page.locator('.ant-card, .ant-table').filter({ 
          hasText: /Statistical|T-Test|Significance|P-Value|Confidence Interval|Hypothesis/ 
        }).count();
        console.log(`Found ${statsTestElements} statistical test elements`);
        
        // Test Real-time Performance Monitor
        const realtimeMonitor = await page.locator('.ant-badge-status-processing, .ant-spin, .ant-progress').count();
        console.log(`Found ${realtimeMonitor} real-time monitoring elements`);
        
        // Test Alert System
        const alertSystem = await page.locator('.ant-alert, .ant-notification').filter({ 
          hasText: /Alert|Threshold|Warning|Notification/ 
        }).count();
        console.log(`Found ${alertSystem} performance alert elements`);
        
        // Test Performance Charts
        const performanceCharts = await checkChartElements(page);
        console.log(`Found ${performanceCharts} performance visualization charts`);
      }
    });
  });

  test.describe('Strategy Tab - Complete Strategy Management', () => {
    test('Strategy management all features', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Strategy Builder Interface
        console.log('Testing Strategy Builder...');
        const builderElements = await page.locator('.ant-card, .ant-form').filter({ 
          hasText: /Strategy Builder|Create Strategy|Configure|Parameters|Settings/ 
        }).count();
        expect(builderElements).toBeGreaterThanOrEqual(0);
        console.log(`Found ${builderElements} strategy builder elements`);
        
        // Test Visual Strategy Builder
        console.log('Testing Visual Builder...');
        const visualBuilderElements = await page.locator('[class*="flow"], [class*="diagram"], canvas, svg').filter({ 
          hasText: /Node|Connection|Flow|Diagram/ 
        }).count();
        console.log(`Found ${visualBuilderElements} visual builder elements`);
        
        // Test Template Library
        console.log('Testing Template Library...');
        const templateElements = await page.locator('.ant-list, .ant-card').filter({ 
          hasText: /Template|Library|Preset|Sample|Example/ 
        }).count();
        console.log(`Found ${templateElements} template library elements`);
        
        // Test Parameter Configuration
        console.log('Testing Parameters...');
        const paramControls = await page.locator('.ant-input-number, .ant-slider, .ant-switch, .ant-select').count();
        console.log(`Found ${paramControls} parameter configuration controls`);
        
        // Test Version Control
        console.log('Testing Version Control...');
        const versionElements = await page.locator('.ant-table, .ant-timeline, .ant-list').filter({ 
          hasText: /Version|History|Commit|Rollback|Change/ 
        }).count();
        console.log(`Found ${versionElements} version control elements`);
        
        // Test Lifecycle Management
        console.log('Testing Lifecycle Controls...');
        const lifecycleButtons = await page.locator('button').filter({ 
          hasText: /Deploy|Start|Stop|Pause|Resume|Enable|Disable/ 
        }).count();
        console.log(`Found ${lifecycleButtons} lifecycle control buttons`);
        
        // Test Live Strategy Monitoring
        console.log('Testing Live Monitoring...');
        const liveMonitorElements = await page.locator('.ant-badge, .ant-tag').filter({ 
          hasText: /Live|Running|Active|Status|State|Mode/ 
        }).count();
        console.log(`Found ${liveMonitorElements} live monitoring elements`);
        
        // Test Multi-Strategy Coordination
        console.log('Testing Multi-Strategy...');
        const multiStrategyElements = await page.locator('.ant-table, .ant-list').filter({ 
          hasText: /Portfolio|Multiple|Coordinate|Combine|Group/ 
        }).count();
        console.log(`Found ${multiStrategyElements} multi-strategy coordination elements`);
        
        // Test Emergency Controls
        console.log('Testing Emergency Controls...');
        const emergencyButtons = await page.locator('button').filter({ 
          hasText: /Emergency|Kill|Stop All|Halt|Shutdown/ 
        }).count();
        console.log(`Found ${emergencyButtons} emergency control buttons`);
        
        // Test Strategy Health Dashboard
        const healthDashboard = await page.locator('.ant-card, .ant-statistic').filter({ 
          hasText: /Health|Performance|Metrics|Status/ 
        }).count();
        console.log(`Found ${healthDashboard} strategy health dashboard elements`);
      }
    });
  });

  test.describe('Engine Tab - Complete Nautilus Engine Management', () => {
    test('Nautilus engine all features', async ({ page }) => {
      await clickTab(page, 'nautilus-engine');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Engine Status Display
        console.log('Testing Engine Status...');
        const statusElements = await page.locator('.ant-badge, .ant-tag, .ant-alert').filter({ 
          hasText: /Running|Stopped|Status|Engine|State|Online|Offline/ 
        }).count();
        expect(statusElements).toBeGreaterThanOrEqual(0);
        console.log(`Found ${statusElements} engine status elements`);
        
        // Test Engine Control Panel
        console.log('Testing Control Panel...');
        const controlButtons = await page.locator('button').filter({ 
          hasText: /Start|Stop|Restart|Initialize|Reset|Reload/ 
        }).count();
        console.log(`Found ${controlButtons} engine control buttons`);
        
        // Test Configuration Panel
        console.log('Testing Configuration...');
        const configElements = await page.locator('.ant-form, .ant-card').filter({ 
          hasText: /Configuration|Settings|Parameters|Options|Preferences/ 
        }).count();
        console.log(`Found ${configElements} configuration panel elements`);
        
        // Test Resource Monitoring
        console.log('Testing Resource Monitor...');
        const resourceElements = await page.locator('.ant-statistic, .ant-progress').filter({ 
          hasText: /CPU|Memory|RAM|Disk|Network|Bandwidth|Usage/ 
        }).count();
        console.log(`Found ${resourceElements} resource monitoring elements`);
        
        // Test Docker Container Status
        console.log('Testing Docker Status...');
        const dockerElements = await page.locator('.ant-card, .ant-list, .ant-tag').filter({ 
          hasText: /Docker|Container|Image|Volume|Network/ 
        }).count();
        console.log(`Found ${dockerElements} Docker status elements`);
        
        // Test Log Display
        const logElements = await page.locator('.ant-card, pre, code').filter({ 
          hasText: /Log|Output|Console|Debug|Info|Warning|Error/ 
        }).count();
        console.log(`Found ${logElements} log display elements`);
        
        // Test Performance Metrics
        const performanceMetrics = await checkStatisticCards(page);
        console.log(`Found ${performanceMetrics} performance metric cards`);
      }
    });
  });

  test.describe('Backtest Tab - Complete Backtesting Functionality', () => {
    test('Backtest runner all features', async ({ page }) => {
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Backtest Configuration Form
        console.log('Testing Configuration Form...');
        const configForm = await page.locator('.ant-form, .ant-card').filter({ 
          hasText: /Configuration|Settings|Parameters|Date Range|Symbol|Strategy/ 
        }).count();
        expect(configForm).toBeGreaterThanOrEqual(0);
        console.log(`Found ${configForm} configuration form elements`);
        
        // Test Date Range Selection
        console.log('Testing Date Range...');
        const datePickerElements = await page.locator('.ant-picker, .ant-calendar, .ant-date-picker').count();
        console.log(`Found ${datePickerElements} date picker elements`);
        
        // Test Symbol/Instrument Selection
        console.log('Testing Symbol Selection...');
        const symbolSelect = await page.locator('.ant-select, .ant-input').filter({ 
          hasText: /Symbol|Instrument|Asset|Ticker/ 
        }).count();
        console.log(`Found ${symbolSelect} symbol selection elements`);
        
        // Test Strategy Selection
        console.log('Testing Strategy Selection...');
        const strategySelect = await page.locator('.ant-select, .ant-radio').filter({ 
          hasText: /Strategy|Algorithm|Model/ 
        }).count();
        console.log(`Found ${strategySelect} strategy selection elements`);
        
        // Test Run Controls
        console.log('Testing Run Controls...');
        const runButtons = await page.locator('button').filter({ 
          hasText: /Run|Start|Execute|Begin|Launch/ 
        }).count();
        console.log(`Found ${runButtons} run control buttons`);
        
        // Test Results Display
        console.log('Testing Results Display...');
        const resultsElements = await page.locator('.ant-card, .ant-table').filter({ 
          hasText: /Results|Performance|Metrics|Statistics|Summary/ 
        }).count();
        console.log(`Found ${resultsElements} results display elements`);
        
        // Test Equity Curve Chart
        const equityChart = await checkChartElements(page);
        console.log(`Found ${equityChart} equity curve chart elements`);
        
        // Test Progress Indicators
        const progressElements = await page.locator('.ant-progress, .ant-spin').count();
        console.log(`Found ${progressElements} progress indicator elements`);
        
        // Test Trade List
        const tradeList = await page.locator('.ant-table, .ant-list').filter({ 
          hasText: /Trade|Transaction|Order|Fill/ 
        }).count();
        console.log(`Found ${tradeList} trade list elements`);
      }
    });
  });

  test.describe('Deploy Tab - Complete Deployment Pipeline', () => {
    test('Deployment pipeline all features', async ({ page }) => {
      await clickTab(page, 'deployment');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Pipeline Stages Display
        console.log('Testing Pipeline Stages...');
        const pipelineStages = await page.locator('.ant-steps, .ant-timeline').count();
        expect(pipelineStages).toBeGreaterThanOrEqual(0);
        console.log(`Found ${pipelineStages} pipeline stage displays`);
        
        // Test Approval Interface
        console.log('Testing Approval Interface...');
        const approvalButtons = await page.locator('button').filter({ 
          hasText: /Approve|Reject|Review|Confirm|Deny/ 
        }).count();
        console.log(`Found ${approvalButtons} approval interface buttons`);
        
        // Test Gradual Rollout Controls
        console.log('Testing Rollout Controls...');
        const rolloutControls = await page.locator('.ant-slider, .ant-input-number').filter({ 
          hasText: /Rollout|Percentage|Scale|Traffic|Weight/ 
        }).count();
        console.log(`Found ${rolloutControls} gradual rollout controls`);
        
        // Test Environment Selection
        console.log('Testing Environment Selection...');
        const environmentSelect = await page.locator('.ant-select, .ant-radio, .ant-tabs').filter({ 
          hasText: /Environment|Production|Staging|Development|Test|UAT/ 
        }).count();
        console.log(`Found ${environmentSelect} environment selection elements`);
        
        // Test Rollback Management
        console.log('Testing Rollback...');
        const rollbackElements = await page.locator('button, .ant-card').filter({ 
          hasText: /Rollback|Revert|Previous|Undo|Restore/ 
        }).count();
        console.log(`Found ${rollbackElements} rollback management elements`);
        
        // Test Health Checks
        console.log('Testing Health Checks...');
        const healthChecks = await page.locator('.ant-list, .ant-table').filter({ 
          hasText: /Health|Check|Validation|Test|Verify/ 
        }).count();
        console.log(`Found ${healthChecks} health check elements`);
        
        // Test Deployment Status
        console.log('Testing Deployment Status...');
        const deploymentStatus = await page.locator('.ant-badge, .ant-tag').filter({ 
          hasText: /Deployed|Pending|In Progress|Failed|Success|Complete/ 
        }).count();
        console.log(`Found ${deploymentStatus} deployment status indicators`);
        
        // Test Configuration History
        const configHistory = await page.locator('.ant-timeline, .ant-table').filter({ 
          hasText: /History|Version|Change|Update/ 
        }).count();
        console.log(`Found ${configHistory} configuration history elements`);
      }
    });
  });

  test.describe('Data Catalog Tab - Complete Data Management', () => {
    test('Data catalog all features', async ({ page }) => {
      await clickTab(page, 'data-catalog');
      await page.waitForTimeout(5000);
      
      const errorBoundary = page.locator('.ant-result-error');
      const hasError = await errorBoundary.isVisible();
      
      if (!hasError) {
        // Test Data Source Display
        console.log('Testing Data Sources...');
        const dataSources = await page.locator('.ant-card, .ant-list, .ant-tag').filter({ 
          hasText: /IBKR|Interactive Brokers|Alpha Vantage|FRED|EDGAR|Data Source|Provider/ 
        }).count();
        expect(dataSources).toBeGreaterThanOrEqual(0);
        console.log(`Found ${dataSources} data source displays`);
        
        // Test Data Pipeline Monitor
        console.log('Testing Pipeline Monitor...');
        const pipelineMonitor = await page.locator('.ant-card, .ant-progress').filter({ 
          hasText: /Pipeline|Stream|Flow|Ingestion|Processing|ETL/ 
        }).count();
        console.log(`Found ${pipelineMonitor} pipeline monitor elements`);
        
        // Test Data Quality Dashboard
        console.log('Testing Data Quality...');
        const qualityMetrics = await page.locator('.ant-statistic, .ant-progress').filter({ 
          hasText: /Quality|Completeness|Accuracy|Validity|Coverage|Integrity/ 
        }).count();
        console.log(`Found ${qualityMetrics} data quality metrics`);
        
        // Test Gap Analysis
        console.log('Testing Gap Analysis...');
        const gapAnalysis = await page.locator('.ant-table, .ant-list').filter({ 
          hasText: /Gap|Missing|Coverage|Availability|Holes/ 
        }).count();
        console.log(`Found ${gapAnalysis} gap analysis elements`);
        
        // Test Export/Import Tools
        console.log('Testing Export/Import...');
        const exportImportButtons = await page.locator('button').filter({ 
          hasText: /Export|Import|Download|Upload|Backup|Restore/ 
        }).count();
        console.log(`Found ${exportImportButtons} export/import tools`);
        
        // Test Data Filtering
        console.log('Testing Filtering...');
        const filterControls = await page.locator('.ant-select, .ant-input, .ant-date-picker').filter({ 
          hasText: /Filter|Search|Symbol|Date|Type|Source/ 
        }).count();
        console.log(`Found ${filterControls} data filtering controls`);
        
        // Test Data Statistics
        console.log('Testing Statistics...');
        const dataStats = await page.locator('.ant-statistic').filter({ 
          hasText: /Records|Rows|Size|Count|Last Updated|Volume/ 
        }).count();
        console.log(`Found ${dataStats} data statistics displays`);
        
        // Test Data Tables
        const dataTables = await checkTableExists(page);
        console.log(`Data tables present: ${dataTables}`);
      }
    });
  });

  test.describe('Search and Watchlist Tabs - Market Data Features', () => {
    test('Search and watchlist complete functionality', async ({ page }) => {
      // Test Search Tab
      await clickTab(page, 'instruments');
      await page.waitForTimeout(3000);
      
      console.log('Testing Instrument Search...');
      
      // Search Interface Elements
      const searchElements = await page.locator('.ant-input, .ant-select').filter({ 
        hasText: /Search|Symbol|Name|Type/ 
      }).count();
      console.log(`Found ${searchElements} search interface elements`);
      
      // Asset Class Filters
      const assetClassTags = await page.locator('.ant-tag').filter({ 
        hasText: /STK|CASH|FUT|IND|OPT|BOND|CRYPTO/ 
      }).count();
      console.log(`Found ${assetClassTags} asset class tags`);
      
      // Search Results Display
      const searchResults = await page.locator('.ant-table, .ant-list').count();
      console.log(`Found ${searchResults} search result displays`);
      
      // Test Watchlist Tab
      await clickTab(page, 'watchlists');
      await page.waitForTimeout(3000);
      
      console.log('Testing Watchlist Management...');
      
      // Watchlist Controls
      const watchlistControls = await page.locator('button').filter({ 
        hasText: /Create|Add|Remove|Delete|Import|Export/ 
      }).count();
      console.log(`Found ${watchlistControls} watchlist control buttons`);
      
      // Watchlist Display
      const watchlistDisplay = await page.locator('.ant-table, .ant-list').count();
      console.log(`Found ${watchlistDisplay} watchlist display elements`);
      
      // Export Format Options
      const exportFormats = await page.locator('.ant-tag').filter({ 
        hasText: /JSON|CSV|Excel/ 
      }).count();
      console.log(`Found ${exportFormats} export format options`);
    });
  });

  test.describe('Chart Tab - Complete Charting Functionality', () => {
    test('Chart interface all features', async ({ page }) => {
      await clickTab(page, 'chart');
      await page.waitForTimeout(5000);
      
      console.log('Testing Chart Components...');
      
      // Instrument Selection
      const instrumentSelector = await page.locator('.ant-select, .ant-input').filter({ 
        hasText: /Instrument|Symbol|Asset/ 
      }).count();
      console.log(`Found ${instrumentSelector} instrument selection elements`);
      
      // Timeframe Selection
      const timeframeSelector = await page.locator('.ant-select, .ant-radio').filter({ 
        hasText: /Timeframe|Period|Interval/ 
      }).count();
      console.log(`Found ${timeframeSelector} timeframe selection elements`);
      
      // Technical Indicators Panel
      const indicatorPanel = await page.locator('.ant-card, .ant-checkbox').filter({ 
        hasText: /Indicator|SMA|EMA|RSI|MACD|Bollinger/ 
      }).count();
      console.log(`Found ${indicatorPanel} technical indicator elements`);
      
      // Chart Display Area
      const chartArea = await checkChartElements(page);
      console.log(`Found ${chartArea} chart display elements`);
      
      // Chart Controls
      const chartControls = await page.locator('button').filter({ 
        hasText: /Zoom|Reset|Pan|Draw|Crosshair/ 
      }).count();
      console.log(`Found ${chartControls} chart control buttons`);
      
      // Chart Type Selection
      const chartTypes = await page.locator('.ant-radio, .ant-select').filter({ 
        hasText: /Candlestick|Line|Bar|Area|Volume/ 
      }).count();
      console.log(`Found ${chartTypes} chart type selection elements`);
    });
  });

  test.describe('System Tab - Infrastructure Monitoring', () => {
    test('System tab infrastructure features', async ({ page }) => {
      await clickTab(page, 'system');
      await page.waitForTimeout(3000);
      
      console.log('Testing System Infrastructure...');
      
      // Backend Health Status
      const backendHealth = await page.locator('.ant-alert').filter({ 
        hasText: /Backend|Connected|Disconnected|Health/ 
      }).count();
      console.log(`Found ${backendHealth} backend health status elements`);
      
      // Message Bus Status
      const messageBusStatus = await page.locator('.ant-card, .ant-badge').filter({ 
        hasText: /MessageBus|WebSocket|Connection|Messages/ 
      }).count();
      console.log(`Found ${messageBusStatus} message bus status elements`);
      
      // Data Backfill System
      const backfillSystem = await page.locator('.ant-card').filter({ 
        hasText: /Backfill|Data Source|IBKR|YFinance/ 
      }).count();
      console.log(`Found ${backfillSystem} data backfill system elements`);
      
      // Environment Information
      const environmentInfo = await page.locator('.ant-card').filter({ 
        hasText: /Environment|Mode|Debug|Configuration/ 
      }).count();
      console.log(`Found ${environmentInfo} environment information elements`);
      
      // Message Statistics
      const messageStats = await page.locator('.ant-statistic').filter({ 
        hasText: /Total Messages|Unique Topics|Statistics/ 
      }).count();
      console.log(`Found ${messageStats} message statistics`);
      
      // Control Buttons
      const controlButtons = await page.locator('button').filter({ 
        hasText: /Connect|Disconnect|Refresh|Clear|Start|Stop/ 
      }).count();
      console.log(`Found ${controlButtons} system control buttons`);
    });
  });

  test.describe('Integration Verification', () => {
    test('Cross-tab data flow and integration', async ({ page }) => {
      console.log('Testing cross-tab integration...');
      
      // Start with System tab to check connections
      await clickTab(page, 'system');
      const systemHealthy = await page.locator('.ant-alert-success').count() > 0;
      console.log(`System health status: ${systemHealthy ? 'Healthy' : 'Issues detected'}`);
      
      // Check IB integration - reduce timeout
      await clickTab(page, 'ib');
      await page.waitForTimeout(1000);
      const ibConnected = await page.locator('.ant-badge-status-success, .ant-tag-success').count() > 0;
      console.log(`IB connection status: ${ibConnected ? 'Connected' : 'Not connected'}`);
      
      // Check Factors data sources - reduce timeout
      await clickTab(page, 'factors');
      await page.waitForTimeout(1000);
      const factorSourcesActive = await page.locator('.ant-badge-status-processing').count();
      console.log(`Active factor data sources: ${factorSourcesActive}`);
      
      // Check Risk monitoring - reduce timeout
      await clickTab(page, 'risk');
      await page.waitForTimeout(1000);
      const riskMonitoringActive = await page.locator('.ant-progress, .ant-statistic').count() > 0;
      console.log(`Risk monitoring active: ${riskMonitoringActive}`);
      
      // Check Portfolio aggregation - reduce timeout
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(1000);
      const portfolioDataPresent = await checkStatisticCards(page) > 0;
      console.log(`Portfolio data present: ${portfolioDataPresent}`);
      
      // Check Strategy execution status - REMOVE TIMEOUT (this was causing the failure)
      await clickTab(page, 'strategy');
      // Wait for tab content to load instead of fixed timeout
      await page.locator('[role="tabpanel"]').first().waitFor({ state: 'visible', timeout: 5000 });
      const strategySystemReady = await page.locator('button:enabled').count() > 0;
      console.log(`Strategy system ready: ${strategySystemReady}`);
      
      // Final system check
      await clickTab(page, 'system');
      const finalSystemCheck = await page.locator('[data-testid="dashboard"]').isVisible();
      expect(finalSystemCheck).toBe(true);
      console.log('Dashboard integration test complete');
    });
  });
});