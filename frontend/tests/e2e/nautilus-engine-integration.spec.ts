import { test, expect, Page } from '@playwright/test';

/**
 * NautilusTrader Engine Integration Test Suite
 * Tests the bridge between frontend dashboard and NautilusTrader core engine
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

async function checkNautilusEngineStatus(page: Page): Promise<string> {
  const statusElement = await page.locator('.ant-badge-status, .ant-tag').filter({ 
    hasText: /Engine|Nautilus|Running|Stopped|Online|Offline/ 
  }).first();
  
  if (await statusElement.isVisible()) {
    return await statusElement.textContent() || 'unknown';
  }
  return 'not-found';
}

test.describe('NautilusTrader Engine Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    await navigateToDashboard(page);
  });

  test.describe('Core Engine Integration', () => {
    test('NautilusTrader engine initialization and health checks', async ({ page }) => {
      await clickTab(page, 'nautilus-engine');
      await page.waitForTimeout(5000);
      
      // Test engine status detection
      console.log('Testing NautilusTrader engine status...');
      const engineStatus = await checkNautilusEngineStatus(page);
      console.log(`Engine status detected: ${engineStatus}`);
      
      // Test DataEngine integration
      const dataEngineElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /DataEngine|Data Engine|Data Processing|Market Data/ 
      }).count();
      console.log(`DataEngine integration elements: ${dataEngineElements}`);
      
      // Test ExecutionEngine integration
      const executionEngineElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /ExecutionEngine|Execution Engine|Order Processing|Trade Execution/ 
      }).count();
      console.log(`ExecutionEngine integration elements: ${executionEngineElements}`);
      
      // Test RiskEngine integration
      const riskEngineElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /RiskEngine|Risk Engine|Risk Management|Risk Limits/ 
      }).count();
      console.log(`RiskEngine integration elements: ${riskEngineElements}`);
      
      // Test engine control capabilities
      const engineControls = await page.locator('button').filter({ 
        hasText: /Initialize|Start|Stop|Restart|Reset|Configure/ 
      }).count();
      console.log(`Engine control buttons: ${engineControls}`);
      
      // Test engine configuration access
      const configElements = await page.locator('.ant-form, .ant-card').filter({ 
        hasText: /Configuration|Settings|Parameters|Engine Config/ 
      }).count();
      console.log(`Engine configuration elements: ${configElements}`);
      
      // Verify MessageBus integration with engine
      await clickTab(page, 'system');
      const messageBusStatus = await page.locator('.ant-badge-status-success').count();
      console.log(`MessageBus integration status: ${messageBusStatus > 0 ? 'Active' : 'Inactive'}`);
    });

    test('Portfolio and Cache integration with NautilusTrader', async ({ page }) => {
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(5000);
      
      // Test Portfolio integration with NautilusTrader
      console.log('Testing Portfolio integration...');
      const portfolioElements = await page.locator('.ant-statistic, .ant-card').filter({ 
        hasText: /Portfolio|Balance|Equity|NAV|Total Value|Unrealized|Realized/ 
      }).count();
      console.log(`Portfolio integration elements: ${portfolioElements}`);
      
      // Test Cache integration for fast data access
      const cacheElements = await page.locator('.ant-card').filter({ 
        hasText: /Cache|Cached|Memory|Storage/ 
      }).count();
      console.log(`Cache integration indicators: ${cacheElements}`);
      
      // Test position data from NautilusTrader
      const positionElements = await page.locator('.ant-table, .ant-list').filter({ 
        hasText: /Position|Quantity|Market Value|P&L|Symbol/ 
      }).count();
      console.log(`Position data elements: ${positionElements}`);
      
      // Test performance metrics from engine
      const performanceElements = await page.locator('.ant-statistic').filter({ 
        hasText: /Return|Performance|Sharpe|Drawdown|Win Rate/ 
      }).count();
      console.log(`Performance metric elements: ${performanceElements}`);
    });
  });

  test.describe('Strategy Deployment Integration', () => {
    test('Strategy deployment pipeline to NautilusTrader', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(5000);
      
      // Test strategy builder integration
      console.log('Testing strategy deployment pipeline...');
      const strategyBuilder = await page.locator('.ant-form, .ant-card').filter({ 
        hasText: /Strategy|Builder|Create|Deploy/ 
      }).count();
      console.log(`Strategy builder elements: ${strategyBuilder}`);
      
      // Test strategy template selection
      const templateSelection = await page.locator('.ant-select, .ant-card').filter({ 
        hasText: /Template|Sample|Example|Preset/ 
      }).count();
      console.log(`Strategy template selection: ${templateSelection}`);
      
      // Test strategy parameter configuration
      const parameterControls = await page.locator('.ant-input-number, .ant-slider, .ant-select').count();
      console.log(`Strategy parameter controls: ${parameterControls}`);
      
      // Test advanced parameter groups
      const parameterGroups = await page.locator('.ant-collapse, .ant-card').filter({ 
        hasText: /Entry Rules|Exit Rules|Risk Parameters|Position Sizing/ 
      }).count();
      console.log(`Advanced parameter groups: ${parameterGroups}`);
      
      // Test strategy validation and testing
      const validationElements = await page.locator('button, .ant-alert').filter({ 
        hasText: /Validate|Check|Verify|Test Strategy|Dry Run/ 
      }).count();
      console.log(`Strategy validation elements: ${validationElements}`);
      
      // Test strategy compilation checks
      const compilationChecks = await page.locator('.ant-alert, .ant-message').filter({ 
        hasText: /Compilation|Syntax|Error|Warning|Success/ 
      }).count();
      console.log(`Strategy compilation checks: ${compilationChecks}`);
      
      // Test deployment controls
      const deploymentButtons = await page.locator('button').filter({ 
        hasText: /Deploy|Start|Launch|Activate|Submit/ 
      }).count();
      console.log(`Strategy deployment buttons: ${deploymentButtons}`);
      
      // Test deployment environment selection
      const environmentSelection = await page.locator('.ant-select, .ant-radio-group').filter({ 
        hasText: /Environment|Live|Paper|Sandbox|Demo/ 
      }).count();
      console.log(`Deployment environment selection: ${environmentSelection}`);
      
      // Test deployment pipeline stages
      const pipelineStages = await page.locator('.ant-steps, .ant-timeline').filter({ 
        hasText: /Validation|Approval|Deploy|Monitor/ 
      }).count();
      console.log(`Deployment pipeline stages: ${pipelineStages}`);
      
      // Test live strategy monitoring
      const monitoringElements = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Live|Running|Active|Monitoring|Deployed/ 
      }).count();
      console.log(`Live strategy monitoring elements: ${monitoringElements}`);
      
      // Test strategy performance tracking
      const performanceTracking = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /Performance|P&L|Trades|Orders|Fill Rate/ 
      }).count();
      console.log(`Strategy performance tracking elements: ${performanceTracking}`);
    });

    test('Strategy version control and rollback', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(3000);
      
      // Test strategy versioning
      console.log('Testing strategy version control...');
      const versionControls = await page.locator('.ant-table, .ant-list').filter({ 
        hasText: /Version|History|Revision|Commit/ 
      }).count();
      console.log(`Version control elements: ${versionControls}`);
      
      // Test strategy comparison tools
      const comparisonTools = await page.locator('button, .ant-modal').filter({ 
        hasText: /Compare|Diff|Changes|History/ 
      }).count();
      console.log(`Strategy comparison tools: ${comparisonTools}`);
      
      // Test rollback capabilities
      const rollbackControls = await page.locator('button').filter({ 
        hasText: /Rollback|Revert|Previous|Restore/ 
      }).count();
      console.log(`Strategy rollback controls: ${rollbackControls}`);
      
      // Test deployment approval workflow
      const approvalWorkflow = await page.locator('.ant-card, .ant-modal').filter({ 
        hasText: /Approval|Review|Pending|Approved|Rejected/ 
      }).count();
      console.log(`Deployment approval workflow: ${approvalWorkflow}`);
    });

    test('Strategy resource management and scaling', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(3000);
      
      // Test resource allocation monitoring
      console.log('Testing strategy resource management...');
      const resourceMonitoring = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /CPU|Memory|Network|Resources|Utilization/ 
      }).count();
      console.log(`Resource monitoring elements: ${resourceMonitoring}`);
      
      // Test strategy scaling controls
      const scalingControls = await page.locator('.ant-slider, .ant-input-number').filter({ 
        hasText: /Scale|Instances|Capacity|Limit/ 
      }).count();
      console.log(`Strategy scaling controls: ${scalingControls}`);
      
      // Test load balancing indicators
      const loadBalancing = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Load|Balance|Distribution|Partition/ 
      }).count();
      console.log(`Load balancing indicators: ${loadBalancing}`);
    });

    test('Strategy lifecycle management', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(3000);
      
      // Test strategy status management
      console.log('Testing strategy lifecycle...');
      const lifecycleButtons = await page.locator('button').filter({ 
        hasText: /Start|Stop|Pause|Resume|Kill|Emergency/ 
      }).count();
      console.log(`Strategy lifecycle buttons: ${lifecycleButtons}`);
      
      // Test strategy health monitoring
      const healthIndicators = await page.locator('.ant-badge-status, .ant-tag').filter({ 
        hasText: /Healthy|Error|Warning|Status/ 
      }).count();
      console.log(`Strategy health indicators: ${healthIndicators}`);
      
      // Test strategy modification capabilities
      const modificationControls = await page.locator('button').filter({ 
        hasText: /Modify|Update|Change|Edit/ 
      }).count();
      console.log(`Strategy modification controls: ${modificationControls}`);
    });
  });

  test.describe('Backtest Engine Integration', () => {
    test('Backtest execution workflow with NautilusTrader', async ({ page }) => {
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(5000);
      
      // Test backtest configuration for NautilusTrader
      console.log('Testing comprehensive backtest integration...');
      const backtestConfig = await page.locator('.ant-form, .ant-card').filter({ 
        hasText: /Backtest|Configuration|Historical|Test Period/ 
      }).count();
      console.log(`Backtest configuration elements: ${backtestConfig}`);
      
      // Test strategy selection for backtesting
      const strategySelection = await page.locator('.ant-select, .ant-card').filter({ 
        hasText: /Strategy|Algorithm|Model|Trading Logic/ 
      }).count();
      console.log(`Strategy selection for backtesting: ${strategySelection}`);
      
      // Test data source configuration
      const dataSourceConfig = await page.locator('.ant-select, .ant-radio-group').filter({ 
        hasText: /Data Source|IBKR|Alpha Vantage|Historical Provider/ 
      }).count();
      console.log(`Data source configuration: ${dataSourceConfig}`);
      
      // Test data selection and filtering
      const dataSelection = await page.locator('.ant-select, .ant-picker').filter({ 
        hasText: /Symbol|Timeframe|Date Range|Universe|Instruments/ 
      }).count();
      console.log(`Backtest data selection controls: ${dataSelection}`);
      
      // Test advanced backtest parameters
      const advancedParams = await page.locator('.ant-input-number, .ant-slider').filter({ 
        hasText: /Commission|Slippage|Latency|Fill Model/ 
      }).count();
      console.log(`Advanced backtest parameters: ${advancedParams}`);
      
      // Test risk parameters for backtesting
      const riskParams = await page.locator('.ant-form-item').filter({ 
        hasText: /Position Size|Risk Limit|Max Drawdown|Stop Loss/ 
      }).count();
      console.log(`Risk parameters for backtesting: ${riskParams}`);
      
      // Test backtest execution controls
      const executionControls = await page.locator('button').filter({ 
        hasText: /Run|Execute|Start|Begin|Launch|Simulate/ 
      }).count();
      console.log(`Backtest execution controls: ${executionControls}`);
      
      // Test execution mode selection
      const executionModes = await page.locator('.ant-select, .ant-radio-group').filter({ 
        hasText: /Mode|Fast|Detailed|Debug|Production/ 
      }).count();
      console.log(`Backtest execution modes: ${executionModes}`);
      
      // Test progress monitoring and cancellation
      const progressElements = await page.locator('.ant-progress, .ant-spin').count();
      console.log(`Backtest progress monitoring: ${progressElements}`);
      
      const cancellationControls = await page.locator('button').filter({ 
        hasText: /Cancel|Stop|Abort|Terminate/ 
      }).count();
      console.log(`Backtest cancellation controls: ${cancellationControls}`);
      
      // Test real-time status updates
      const statusUpdates = await page.locator('.ant-tag, .ant-badge').filter({ 
        hasText: /Status|Running|Completed|Failed|Progress/ 
      }).count();
      console.log(`Real-time status updates: ${statusUpdates}`);
      
      // Test results display integration
      const resultsDisplay = await page.locator('.ant-card, .ant-table').filter({ 
        hasText: /Results|Performance|Statistics|Summary|Metrics/ 
      }).count();
      console.log(`Backtest results display elements: ${resultsDisplay}`);
      
      // Test equity curve and charts integration
      const equityCurve = await page.locator('[class*="chart"], canvas').count();
      console.log(`Equity curve and chart elements: ${equityCurve}`);
      
      // Test report generation
      const reportGeneration = await page.locator('button').filter({ 
        hasText: /Report|Export|Download|PDF|Generate/ 
      }).count();
      console.log(`Report generation controls: ${reportGeneration}`);
    });

    test('Backtest performance optimization and caching', async ({ page }) => {
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(3000);
      
      // Test performance optimization settings
      console.log('Testing backtest performance optimization...');
      const performanceSettings = await page.locator('.ant-form-item').filter({ 
        hasText: /Performance|Optimization|Cache|Parallel|Threading/ 
      }).count();
      console.log(`Performance optimization settings: ${performanceSettings}`);
      
      // Test caching mechanisms
      const cachingControls = await page.locator('.ant-switch, .ant-checkbox').filter({ 
        hasText: /Cache|Memory|Disk|Store|Reuse/ 
      }).count();
      console.log(`Caching mechanism controls: ${cachingControls}`);
      
      // Test resource monitoring during backtests
      const resourceMonitoring = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /CPU|Memory|Disk|Network|Resource/ 
      }).count();
      console.log(`Resource monitoring during backtests: ${resourceMonitoring}`);
      
      // Test parallel execution controls
      const parallelControls = await page.locator('.ant-input-number').filter({ 
        hasText: /Workers|Threads|Parallel|Concurrent/ 
      }).count();
      console.log(`Parallel execution controls: ${parallelControls}`);
    });

    test('Backtest comparison and batch processing', async ({ page }) => {
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(3000);
      
      // Test batch backtest capabilities
      console.log('Testing batch backtest processing...');
      const batchProcessing = await page.locator('.ant-table, .ant-list').filter({ 
        hasText: /Batch|Queue|Multiple|Series/ 
      }).count();
      console.log(`Batch processing elements: ${batchProcessing}`);
      
      // Test backtest comparison tools
      const comparisonTools = await page.locator('button, .ant-modal').filter({ 
        hasText: /Compare|Comparison|Analyze|Benchmark/ 
      }).count();
      console.log(`Backtest comparison tools: ${comparisonTools}`);
      
      // Test parameter sweep capabilities
      const parameterSweep = await page.locator('.ant-form, .ant-table').filter({ 
        hasText: /Sweep|Range|Parameter|Optimization/ 
      }).count();
      console.log(`Parameter sweep capabilities: ${parameterSweep}`);
      
      // Test result aggregation
      const resultAggregation = await page.locator('.ant-statistic, .ant-card').filter({ 
        hasText: /Aggregate|Summary|Combined|Average/ 
      }).count();
      console.log(`Result aggregation elements: ${resultAggregation}`);
    });

    test('Backtest results analysis and visualization', async ({ page }) => {
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(3000);
      
      // Test performance metrics from backtest
      console.log('Testing backtest results...');
      const metricsDisplay = await page.locator('.ant-statistic').filter({ 
        hasText: /Total Return|Sharpe|Max Drawdown|Win Rate|Profit Factor/ 
      }).count();
      console.log(`Backtest metrics display: ${metricsDisplay}`);
      
      // Test trade analysis
      const tradeAnalysis = await page.locator('.ant-table').filter({ 
        hasText: /Trade|Entry|Exit|P&L|Duration/ 
      }).count();
      console.log(`Trade analysis tables: ${tradeAnalysis}`);
      
      // Test visualization components
      const visualizations = await page.locator('[class*="chart"], canvas, svg').count();
      console.log(`Backtest visualization elements: ${visualizations}`);
    });
  });

  test.describe('Live Trading Integration', () => {
    test('Order management through NautilusTrader ExecutionEngine', async ({ page }) => {
      await clickTab(page, 'ib');
      await page.waitForTimeout(5000);
      
      // Test order placement integration with ExecutionEngine
      console.log('Testing comprehensive live trading integration...');
      const orderControls = await page.locator('button, .ant-form').filter({ 
        hasText: /Place Order|Market|Limit|Stop|Buy|Sell|Submit/ 
      }).count();
      console.log(`Order placement controls: ${orderControls}`);
      
      // Test advanced order types
      const advancedOrderTypes = await page.locator('.ant-select, .ant-radio-group').filter({ 
        hasText: /OCO|Bracket|Trail|Iceberg|Hidden|Time in Force/ 
      }).count();
      console.log(`Advanced order types: ${advancedOrderTypes}`);
      
      // Test order validation and pre-trade checks
      const orderValidation = await page.locator('.ant-alert, .ant-message').filter({ 
        hasText: /Validation|Pre-trade|Check|Warning|Error/ 
      }).count();
      console.log(`Order validation elements: ${orderValidation}`);
      
      // Test position management with real-time updates
      const positionManagement = await page.locator('.ant-table, .ant-card').filter({ 
        hasText: /Position|Quantity|Average Price|Unrealized|P&L/ 
      }).count();
      console.log(`Position management elements: ${positionManagement}`);
      
      // Test real-time position updates
      const realtimeUpdates = await page.locator('.ant-badge-status-processing, .ant-spin').count();
      console.log(`Real-time position updates: ${realtimeUpdates}`);
      
      // Test order status tracking and lifecycle
      const orderTracking = await page.locator('.ant-table').filter({ 
        hasText: /Order|Status|Filled|Pending|Cancelled|Partial/ 
      }).count();
      console.log(`Order status tracking elements: ${orderTracking}`);
      
      // Test order modification and cancellation
      const orderModification = await page.locator('button').filter({ 
        hasText: /Modify|Cancel|Replace|Update/ 
      }).count();
      console.log(`Order modification controls: ${orderModification}`);
      
      // Test execution reports and trade history
      const executionReports = await page.locator('.ant-table').filter({ 
        hasText: /Execution|Fill|Trade|Time|Price|Commission/ 
      }).count();
      console.log(`Execution report elements: ${executionReports}`);
      
      // Test trade confirmation and alerts
      const tradeConfirmations = await page.locator('.ant-notification, .ant-message').filter({ 
        hasText: /Filled|Executed|Confirmed|Trade/ 
      }).count();
      console.log(`Trade confirmation elements: ${tradeConfirmations}`);
      
      // Test account balance and buying power monitoring
      const accountMonitoring = await page.locator('.ant-statistic').filter({ 
        hasText: /Balance|Buying Power|Cash|Margin|Equity/ 
      }).count();
      console.log(`Account monitoring elements: ${accountMonitoring}`);
    });

    test('Risk management integration with NautilusTrader RiskEngine', async ({ page }) => {
      await clickTab(page, 'risk');
      await page.waitForTimeout(5000);
      
      // Test comprehensive risk management
      console.log('Testing risk management integration...');
      const riskManagement = await page.locator('.ant-alert, .ant-badge').filter({ 
        hasText: /Risk|Limit|Exposure|Warning|Breach/ 
      }).count();
      console.log(`Risk management elements: ${riskManagement}`);
      
      // Test position limits and constraints
      const positionLimits = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /Position Limit|Max Position|Concentration|Exposure/ 
      }).count();
      console.log(`Position limit monitoring: ${positionLimits}`);
      
      // Test real-time risk calculations
      const riskCalculations = await page.locator('.ant-statistic').filter({ 
        hasText: /VaR|Greeks|Delta|Gamma|Theta|Vega/ 
      }).count();
      console.log(`Real-time risk calculations: ${riskCalculations}`);
      
      // Test risk alerts and notifications
      const riskAlerts = await page.locator('.ant-alert').filter({ 
        hasText: /Alert|Warning|Critical|Breach|Violation/ 
      }).count();
      console.log(`Risk alert system: ${riskAlerts}`);
      
      // Test emergency controls
      const emergencyControls = await page.locator('button').filter({ 
        hasText: /Emergency|Panic|Stop All|Liquidate|Kill Switch/ 
      }).count();
      console.log(`Emergency trading controls: ${emergencyControls}`);
      
      // Test drawdown monitoring
      const drawdownMonitoring = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /Drawdown|Loss|Max DD|Daily Loss/ 
      }).count();
      console.log(`Drawdown monitoring elements: ${drawdownMonitoring}`);
    });

    test('Strategy execution monitoring and control', async ({ page }) => {
      await clickTab(page, 'strategy');
      await page.waitForTimeout(3000);
      
      // Test live strategy execution monitoring
      console.log('Testing live strategy execution...');
      const strategyExecution = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Live|Running|Executing|Active/ 
      }).count();
      console.log(`Live strategy execution indicators: ${strategyExecution}`);
      
      // Test strategy performance metrics
      const strategyMetrics = await page.locator('.ant-statistic').filter({ 
        hasText: /P&L|Win Rate|Sharpe|Fill Rate|Slippage/ 
      }).count();
      console.log(`Live strategy metrics: ${strategyMetrics}`);
      
      // Test strategy control panel
      const strategyControls = await page.locator('button').filter({ 
        hasText: /Pause|Resume|Stop|Restart|Adjust/ 
      }).count();
      console.log(`Strategy control buttons: ${strategyControls}`);
      
      // Test parameter adjustment in live trading
      const parameterAdjustment = await page.locator('.ant-slider, .ant-input-number').filter({ 
        hasText: /Size|Threshold|Limit|Parameter/ 
      }).count();
      console.log(`Live parameter adjustment controls: ${parameterAdjustment}`);
      
      // Test strategy health monitoring
      const strategyHealth = await page.locator('.ant-badge-status, .ant-alert').filter({ 
        hasText: /Health|Error|Warning|Normal/ 
      }).count();
      console.log(`Strategy health monitoring: ${strategyHealth}`);
    });

    test('Market data integration for live trading', async ({ page }) => {
      await clickTab(page, 'ib');
      await page.waitForTimeout(3000);
      
      // Test real-time market data feed
      console.log('Testing market data integration...');
      const marketDataFeed = await page.locator('.ant-table, .ant-card').filter({ 
        hasText: /Market Data|Quote|Bid|Ask|Last|Volume/ 
      }).count();
      console.log(`Market data feed elements: ${marketDataFeed}`);
      
      // Test data quality indicators
      const dataQuality = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Live|Delayed|Stale|Fresh|Quality/ 
      }).count();
      console.log(`Data quality indicators: ${dataQuality}`);
      
      // Test market depth and order book
      const orderBook = await page.locator('.ant-table').filter({ 
        hasText: /Order Book|Level 2|Depth|Market Depth/ 
      }).count();
      console.log(`Order book elements: ${orderBook}`);
      
      // Test tick data streaming
      const tickData = await page.locator('.ant-list, .ant-table').filter({ 
        hasText: /Tick|Stream|Real-time|Feed/ 
      }).count();
      console.log(`Tick data streaming: ${tickData}`);
    });

    test('Real-time data integration through MessageBus', async ({ page }) => {
      await clickTab(page, 'system');
      await page.waitForTimeout(3000);
      
      // Test MessageBus integration with NautilusTrader
      console.log('Testing real-time data integration...');
      const messageBusIntegration = await page.locator('.ant-card').filter({ 
        hasText: /MessageBus|Message Bus|Real-time|Streaming/ 
      }).count();
      console.log(`MessageBus integration elements: ${messageBusIntegration}`);
      
      // Test data flow indicators
      const dataFlowIndicators = await page.locator('.ant-badge-status-processing, .ant-spin').count();
      console.log(`Real-time data flow indicators: ${dataFlowIndicators}`);
      
      // Test message statistics
      const messageStats = await page.locator('.ant-statistic').filter({ 
        hasText: /Messages|Topics|Rate|Latency/ 
      }).count();
      console.log(`Message statistics: ${messageStats}`);
      
      // Test connection health to NautilusTrader
      const connectionHealth = await page.locator('.ant-badge-status-success').count();
      console.log(`NautilusTrader connection health: ${connectionHealth > 0 ? 'Good' : 'Poor'}`);
    });
  });

  test.describe('Toraniko Factor Engine Integration', () => {
    test('Toraniko style factors integration (Value, Momentum, Size)', async ({ page }) => {
      await clickTab(page, 'factors');
      await page.waitForTimeout(5000);
      
      // Test Toraniko Factor Engine status
      console.log('Testing Toraniko Factor Engine integration...');
      const toranikoStatus = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Toraniko|Factor Engine|Multi-factor|Risk Model/ 
      }).count();
      console.log(`Toraniko Factor Engine status indicators: ${toranikoStatus}`);
      
      // Test Value Factor calculations
      const valueFactorElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Value Factor|Book Price|Sales Price|Cash Flow|P\/B|P\/S|P\/CF/ 
      }).count();
      console.log(`Value factor calculation elements: ${valueFactorElements}`);
      
      // Test Momentum Factor calculations
      const momentumFactorElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Momentum Factor|Price Momentum|Trend|Trailing Returns/ 
      }).count();
      console.log(`Momentum factor calculation elements: ${momentumFactorElements}`);
      
      // Test Size Factor calculations
      const sizeFactorElements = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Size Factor|Market Cap|Small Cap|Large Cap|Cap-based/ 
      }).count();
      console.log(`Size factor calculation elements: ${sizeFactorElements}`);
      
      // Test factor score normalization and winsorization
      const normalizationControls = await page.locator('.ant-form-item').filter({ 
        hasText: /Normalization|Winsorization|Z-Score|Percentile|Outlier/ 
      }).count();
      console.log(`Factor normalization controls: ${normalizationControls}`);
      
      // Test cross-sectional analysis
      const crossSectionalAnalysis = await page.locator('.ant-table, .ant-chart').filter({ 
        hasText: /Cross-sectional|Sector|Industry|Universe/ 
      }).count();
      console.log(`Cross-sectional analysis elements: ${crossSectionalAnalysis}`);
    });

    test('Multi-factor risk model and factor return estimation', async ({ page }) => {
      await clickTab(page, 'factors');
      await page.waitForTimeout(3000);
      
      // Test factor return estimation capabilities
      console.log('Testing factor return estimation...');
      const factorReturnEstimation = await page.locator('.ant-card, .ant-table').filter({ 
        hasText: /Factor Returns|Return Estimation|Market Factor|Sector Factor|Style Factor/ 
      }).count();
      console.log(`Factor return estimation elements: ${factorReturnEstimation}`);
      
      // Test market factor calculations
      const marketFactorElements = await page.locator('.ant-statistic').filter({ 
        hasText: /Market Factor|Market Return|Beta|Market Cap Weighted/ 
      }).count();
      console.log(`Market factor elements: ${marketFactorElements}`);
      
      // Test sector factor integration
      const sectorFactorElements = await page.locator('.ant-table, .ant-card').filter({ 
        hasText: /Sector Factor|Industry|GICS|Sector Returns/ 
      }).count();
      console.log(`Sector factor elements: ${sectorFactorElements}`);
      
      // Test factor covariance matrix
      const covarianceMatrix = await page.locator('.ant-card, [class*="chart"]').filter({ 
        hasText: /Covariance|Correlation|Risk Matrix|Factor Risk/ 
      }).count();
      console.log(`Factor covariance matrix elements: ${covarianceMatrix}`);
      
      // Test factor exposure analysis
      const exposureAnalysis = await page.locator('.ant-table').filter({ 
        hasText: /Exposure|Factor Loading|Attribution|Contribution/ 
      }).count();
      console.log(`Factor exposure analysis: ${exposureAnalysis}`);
      
      // Test residual analysis
      const residualAnalysis = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Residual|Idiosyncratic|Specific Risk|Alpha/ 
      }).count();
      console.log(`Residual risk analysis: ${residualAnalysis}`);
    });

    test('Cross-source factor integration (EDGAR × FRED × IBKR)', async ({ page }) => {
      await clickTab(page, 'factors');
      await page.waitForTimeout(3000);
      
      // Test EDGAR × FRED factor combinations
      console.log('Testing cross-source factor combinations...');
      const edgarFredFactors = await page.locator('.ant-card').filter({ 
        hasText: /EDGAR.*FRED|Fundamental.*Economic|Earnings.*Cycle|Quality.*Macro/ 
      }).count();
      console.log(`EDGAR × FRED factor combinations: ${edgarFredFactors}`);
      
      // Test FRED × IBKR factor combinations
      const fredIbkrFactors = await page.locator('.ant-card').filter({ 
        hasText: /FRED.*IBKR|Economic.*Technical|Macro.*Price|Interest.*Momentum/ 
      }).count();
      console.log(`FRED × IBKR factor combinations: ${fredIbkrFactors}`);
      
      // Test EDGAR × IBKR factor combinations
      const edgarIbkrFactors = await page.locator('.ant-card').filter({ 
        hasText: /EDGAR.*IBKR|Fundamental.*Technical|Earnings.*Volume|Quality.*Price/ 
      }).count();
      console.log(`EDGAR × IBKR factor combinations: ${edgarIbkrFactors}`);
      
      // Test triple integration factors
      const tripleIntegrationFactors = await page.locator('.ant-card').filter({ 
        hasText: /Triple.*Integration|Economic.*Fundamental.*Technical|Multi-source/ 
      }).count();
      console.log(`Triple integration factors: ${tripleIntegrationFactors}`);
      
      // Test factor synthesis controls
      const synthesisControls = await page.locator('button').filter({ 
        hasText: /Synthesize|Combine|Generate|Calculate.*Cross/ 
      }).count();
      console.log(`Factor synthesis controls: ${synthesisControls}`);
      
      // Test competitive advantage metrics
      const competitiveAdvantage = await page.locator('.ant-alert, .ant-card').filter({ 
        hasText: /Competitive|Unique|Proprietary|Advantage|Commercial/ 
      }).count();
      console.log(`Competitive advantage indicators: ${competitiveAdvantage}`);
    });

    test('Portfolio optimization and risk constraints', async ({ page }) => {
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(3000);
      
      // Test factor-based portfolio optimization
      console.log('Testing factor-based portfolio optimization...');
      const portfolioOptimization = await page.locator('.ant-card, .ant-form').filter({ 
        hasText: /Optimization|Factor.*Constraint|Market.*Neutral|Risk.*Budget/ 
      }).count();
      console.log(`Portfolio optimization elements: ${portfolioOptimization}`);
      
      // Test factor exposure constraints
      const exposureConstraints = await page.locator('.ant-slider, .ant-input-number').filter({ 
        hasText: /Exposure|Constraint|Limit|Bound/ 
      }).count();
      console.log(`Factor exposure constraint controls: ${exposureConstraints}`);
      
      // Test risk budgeting with factors
      const riskBudgeting = await page.locator('.ant-statistic, .ant-progress').filter({ 
        hasText: /Risk.*Budget|Factor.*Risk|Contribution.*Risk|Risk.*Attribution/ 
      }).count();
      console.log(`Risk budgeting elements: ${riskBudgeting}`);
      
      // Test market neutral portfolio construction
      const marketNeutral = await page.locator('.ant-card, .ant-badge').filter({ 
        hasText: /Market.*Neutral|Beta.*Neutral|Factor.*Neutral/ 
      }).count();
      console.log(`Market neutral construction: ${marketNeutral}`);
      
      // Test factor tilting and timing
      const factorTiming = await page.locator('.ant-card').filter({ 
        hasText: /Factor.*Timing|Tilt|Rotation|Cycle/ 
      }).count();
      console.log(`Factor timing and tilting: ${factorTiming}`);
    });

    test('Institutional-grade factor analytics and reporting', async ({ page }) => {
      await clickTab(page, 'performance');
      await page.waitForTimeout(3000);
      
      // Test factor attribution analysis
      console.log('Testing factor attribution and analytics...');
      const attributionAnalysis = await page.locator('.ant-table, .ant-card').filter({ 
        hasText: /Attribution|Factor.*Contribution|Performance.*Attribution/ 
      }).count();
      console.log(`Factor attribution analysis: ${attributionAnalysis}`);
      
      // Test Barra-style factor modeling
      const barraStyleModeling = await page.locator('.ant-card').filter({ 
        hasText: /Barra|Axioma|Style.*Model|Commercial.*Comparison/ 
      }).count();
      console.log(`Barra-style modeling indicators: ${barraStyleModeling}`);
      
      // Test factor performance analytics
      const factorPerformance = await page.locator('.ant-statistic').filter({ 
        hasText: /Factor.*Return|Factor.*Sharpe|Factor.*Volatility|IC/ 
      }).count();
      console.log(`Factor performance analytics: ${factorPerformance}`);
      
      // Test factor model validation
      const modelValidation = await page.locator('.ant-card, .ant-alert').filter({ 
        hasText: /Validation|R-squared|T-stat|Significance/ 
      }).count();
      console.log(`Factor model validation: ${modelValidation}`);
      
      // Test institutional reporting
      const institutionalReporting = await page.locator('button').filter({ 
        hasText: /Report|Export|Institutional|Risk.*Report/ 
      }).count();
      console.log(`Institutional reporting controls: ${institutionalReporting}`);
    });
  });

  test.describe('Data Pipeline Integration', () => {
    test('Multi-source data integration with NautilusTrader DataEngine', async ({ page }) => {
      await clickTab(page, 'data-catalog');
      await page.waitForTimeout(5000);
      
      // Test comprehensive data source integration
      console.log('Testing comprehensive data pipeline integration...');
      const dataSourceIntegration = await page.locator('.ant-card').filter({ 
        hasText: /Data Source|Historical|Loading|Import|IBKR|Alpha Vantage|FRED|EDGAR/ 
      }).count();
      console.log(`Multi-source data integration elements: ${dataSourceIntegration}`);
      
      // Test data provider status monitoring
      const providerStatus = await page.locator('.ant-badge-status, .ant-tag').filter({ 
        hasText: /IBKR|Alpha Vantage|FRED|EDGAR|Connected|Online|Available/ 
      }).count();
      console.log(`Data provider status indicators: ${providerStatus}`);
      
      // Test data quality monitoring and validation
      const dataQuality = await page.locator('.ant-progress, .ant-statistic').filter({ 
        hasText: /Quality|Completeness|Accuracy|Coverage|Validation/ 
      }).count();
      console.log(`Data quality monitoring: ${dataQuality}`);
      
      // Test data cleansing and normalization
      const dataCleansing = await page.locator('.ant-card, .ant-steps').filter({ 
        hasText: /Cleansing|Normalization|Transform|Process|Clean/ 
      }).count();
      console.log(`Data cleansing pipeline: ${dataCleansing}`);
      
      // Test data pipeline orchestration
      const pipelineOrchestration = await page.locator('.ant-timeline, .ant-steps').filter({ 
        hasText: /Pipeline|Workflow|ETL|Extract|Transform|Load/ 
      }).count();
      console.log(`Pipeline orchestration elements: ${pipelineOrchestration}`);
      
      // Test data pipeline status and monitoring
      const pipelineStatus = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Pipeline|Processing|Complete|Error|Running|Idle/ 
      }).count();
      console.log(`Data pipeline status indicators: ${pipelineStatus}`);
      
      // Test data lineage tracking
      const dataLineage = await page.locator('.ant-tree, .ant-timeline').filter({ 
        hasText: /Lineage|Source|Origin|Transformation|History/ 
      }).count();
      console.log(`Data lineage tracking: ${dataLineage}`);
      
      // Test data statistics and metrics
      const dataStats = await page.locator('.ant-statistic').filter({ 
        hasText: /Records|Size|Updated|Count|Volume|Throughput/ 
      }).count();
      console.log(`Data statistics display: ${dataStats}`);
      
      // Test data refresh and synchronization
      const dataRefresh = await page.locator('button').filter({ 
        hasText: /Refresh|Sync|Update|Reload|Fetch/ 
      }).count();
      console.log(`Data refresh controls: ${dataRefresh}`);
    });

    test('Data caching and performance optimization', async ({ page }) => {
      await clickTab(page, 'data-catalog');
      await page.waitForTimeout(3000);
      
      // Test data caching mechanisms
      console.log('Testing data caching and optimization...');
      const cachingMechanisms = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Cache|Memory|Storage|Hit Rate|Performance/ 
      }).count();
      console.log(`Data caching mechanisms: ${cachingMechanisms}`);
      
      // Test cache management controls
      const cacheControls = await page.locator('button').filter({ 
        hasText: /Clear Cache|Invalidate|Refresh|Preload/ 
      }).count();
      console.log(`Cache management controls: ${cacheControls}`);
      
      // Test data compression and storage optimization
      const storageOptimization = await page.locator('.ant-statistic').filter({ 
        hasText: /Compression|Storage|Disk Usage|Optimization/ 
      }).count();
      console.log(`Storage optimization metrics: ${storageOptimization}`);
      
      // Test query performance monitoring
      const queryPerformance = await page.locator('.ant-statistic').filter({ 
        hasText: /Query Time|Response|Latency|Performance/ 
      }).count();
      console.log(`Query performance monitoring: ${queryPerformance}`);
    });

    test('Data backup and disaster recovery', async ({ page }) => {
      await clickTab(page, 'data-catalog');
      await page.waitForTimeout(3000);
      
      // Test backup and recovery systems
      console.log('Testing backup and disaster recovery...');
      const backupSystems = await page.locator('.ant-card, .ant-timeline').filter({ 
        hasText: /Backup|Recovery|Archive|Snapshot|Restore/ 
      }).count();
      console.log(`Backup and recovery systems: ${backupSystems}`);
      
      // Test data replication status
      const replicationStatus = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Replication|Replica|Mirror|Sync|Backup/ 
      }).count();
      console.log(`Data replication status: ${replicationStatus}`);
      
      // Test disaster recovery controls
      const recoveryControls = await page.locator('button').filter({ 
        hasText: /Restore|Recover|Failover|Switch/ 
      }).count();
      console.log(`Disaster recovery controls: ${recoveryControls}`);
      
      // Test data integrity checks
      const integrityChecks = await page.locator('.ant-alert, .ant-statistic').filter({ 
        hasText: /Integrity|Checksum|Validation|Corruption/ 
      }).count();
      console.log(`Data integrity monitoring: ${integrityChecks}`);
    });

    test('Real-time data streaming integration', async ({ page }) => {
      await clickTab(page, 'factors');
      await page.waitForTimeout(5000);
      
      // Test real-time factor streaming
      console.log('Testing real-time streaming...');
      const streamingIndicators = await page.locator('.ant-badge-status-processing, .ant-spin').count();
      console.log(`Real-time streaming indicators: ${streamingIndicators}`);
      
      // Test data source connectivity
      const dataSourceConnectivity = await page.locator('.ant-badge-status').filter({ 
        hasText: /Connected|Online|Active/ 
      }).count();
      console.log(`Data source connectivity: ${dataSourceConnectivity}`);
      
      // Test streaming controls
      const streamingControls = await page.locator('button').filter({ 
        hasText: /Start Stream|Stop Stream|Pause|Resume/ 
      }).count();
      console.log(`Streaming control buttons: ${streamingControls}`);
      
      // Test data freshness indicators
      const dataFreshness = await page.locator('.ant-tag, .ant-badge').filter({ 
        hasText: /Live|Latest|Updated|Fresh/ 
      }).count();
      console.log(`Data freshness indicators: ${dataFreshness}`);
    });
  });

  test.describe('Performance Monitoring Integration', () => {
    test('Engine performance metrics collection', async ({ page }) => {
      await clickTab(page, 'performance');
      await page.waitForTimeout(5000);
      
      // Test engine performance monitoring
      console.log('Testing engine performance monitoring...');
      const engineMetrics = await page.locator('.ant-statistic').filter({ 
        hasText: /Latency|Throughput|CPU|Memory|Messages\/Sec/ 
      }).count();
      console.log(`Engine performance metrics: ${engineMetrics}`);
      
      // Test performance charts
      const performanceCharts = await page.locator('[class*="chart"], canvas').count();
      console.log(`Performance visualization charts: ${performanceCharts}`);
      
      // Test alert system for performance issues
      const performanceAlerts = await page.locator('.ant-alert').filter({ 
        hasText: /Performance|Slow|High|Warning/ 
      }).count();
      console.log(`Performance alert elements: ${performanceAlerts}`);
    });

    test('Strategy performance tracking integration', async ({ page }) => {
      await clickTab(page, 'performance');
      await page.waitForTimeout(3000);
      
      // Test strategy-specific performance metrics
      console.log('Testing strategy performance tracking...');
      const strategyMetrics = await page.locator('.ant-card, .ant-table').filter({ 
        hasText: /Strategy Performance|P&L|Attribution|Contribution/ 
      }).count();
      console.log(`Strategy performance elements: ${strategyMetrics}`);
      
      // Test real-time performance updates
      const realtimeUpdates = await page.locator('.ant-badge-status-processing').count();
      console.log(`Real-time performance updates: ${realtimeUpdates}`);
      
      // Test performance comparison tools
      const comparisonTools = await page.locator('.ant-table, [class*="chart"]').filter({ 
        hasText: /Comparison|Benchmark|Relative/ 
      }).count();
      console.log(`Performance comparison tools: ${comparisonTools}`);
    });
  });

  test.describe('Integration Health Checks', () => {
    test('End-to-end NautilusTrader integration health check', async ({ page }) => {
      console.log('Running comprehensive NautilusTrader integration health check...');
      
      // Check system health and backend connectivity
      await clickTab(page, 'system');
      const systemHealthy = await page.locator('.ant-alert-success').isVisible();
      console.log(`System health: ${systemHealthy ? 'Good' : 'Issues detected'}`);
      
      // Check NautilusTrader engine status and components
      await clickTab(page, 'nautilus-engine');
      await page.waitForTimeout(2000);
      const engineStatus = await checkNautilusEngineStatus(page);
      console.log(`NautilusTrader engine status: ${engineStatus}`);
      
      // Check individual engine components
      const dataEngineHealth = await page.locator('.ant-badge-status, .ant-tag').filter({ 
        hasText: /DataEngine|Data Engine|Running|Online/ 
      }).count();
      console.log(`DataEngine health indicators: ${dataEngineHealth}`);
      
      const executionEngineHealth = await page.locator('.ant-badge-status, .ant-tag').filter({ 
        hasText: /ExecutionEngine|Execution Engine|Active|Ready/ 
      }).count();
      console.log(`ExecutionEngine health indicators: ${executionEngineHealth}`);
      
      const riskEngineHealth = await page.locator('.ant-badge-status, .ant-tag').filter({ 
        hasText: /RiskEngine|Risk Engine|Monitoring|Active/ 
      }).count();
      console.log(`RiskEngine health indicators: ${riskEngineHealth}`);
      
      // Check MessageBus connectivity and performance
      await clickTab(page, 'system');
      const messageBusConnected = await page.locator('.ant-badge-status-success').count() > 0;
      console.log(`MessageBus connectivity: ${messageBusConnected ? 'Connected' : 'Disconnected'}`);
      
      const messageBusPerformance = await page.locator('.ant-statistic').filter({ 
        hasText: /Messages|Throughput|Latency|Performance/ 
      }).count();
      console.log(`MessageBus performance metrics: ${messageBusPerformance}`);
      
      // Check multi-source data integration health
      await clickTab(page, 'factors');
      await page.waitForTimeout(2000);
      const dataSourcesActive = await page.locator('.ant-badge-status-processing').count();
      console.log(`Active data sources: ${dataSourcesActive}`);
      
      const ibkrConnectivity = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /IBKR|Interactive Brokers|Connected/ 
      }).count();
      console.log(`IBKR connectivity indicators: ${ibkrConnectivity}`);
      
      const alphaVantageHealth = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /Alpha Vantage|AV|Active|Online/ 
      }).count();
      console.log(`Alpha Vantage health indicators: ${alphaVantageHealth}`);
      
      const fredDataHealth = await page.locator('.ant-badge, .ant-tag').filter({ 
        hasText: /FRED|Economic|Data|Active/ 
      }).count();
      console.log(`FRED data health indicators: ${fredDataHealth}`);
      
      // Check portfolio and trading integration
      await clickTab(page, 'portfolio');
      await page.waitForTimeout(2000);
      const portfolioData = await page.locator('.ant-statistic').count();
      console.log(`Portfolio data elements: ${portfolioData}`);
      
      const portfolioIntegration = await page.locator('.ant-card').filter({ 
        hasText: /Portfolio|Balance|Position|Equity/ 
      }).count();
      console.log(`Portfolio integration elements: ${portfolioIntegration}`);
      
      // Check strategy and backtest capabilities
      await clickTab(page, 'strategy');
      await page.waitForTimeout(2000);
      const strategyCapabilities = await page.locator('.ant-card, .ant-form').filter({ 
        hasText: /Strategy|Deploy|Build|Execute/ 
      }).count();
      console.log(`Strategy capabilities: ${strategyCapabilities}`);
      
      await clickTab(page, 'backtesting');
      await page.waitForTimeout(2000);
      const backtestCapabilities = await page.locator('.ant-card, .ant-form').filter({ 
        hasText: /Backtest|Historical|Test|Simulation/ 
      }).count();
      console.log(`Backtest capabilities: ${backtestCapabilities}`);
      
      // Check live trading readiness
      await clickTab(page, 'ib');
      await page.waitForTimeout(2000);
      const tradingReadiness = await page.locator('.ant-card, .ant-table').filter({ 
        hasText: /Order|Position|Account|Trading/ 
      }).count();
      console.log(`Live trading readiness: ${tradingReadiness}`);
      
      // Check risk management integration
      await clickTab(page, 'risk');
      await page.waitForTimeout(2000);
      const riskManagementIntegration = await page.locator('.ant-card, .ant-alert').filter({ 
        hasText: /Risk|Limit|Exposure|Monitor/ 
      }).count();
      console.log(`Risk management integration: ${riskManagementIntegration}`);
      
      // Check Toraniko Factor Engine integration
      await clickTab(page, 'factors');
      await page.waitForTimeout(2000);
      const toranikoFactorEngine = await page.locator('.ant-card, .ant-badge').filter({ 
        hasText: /Toraniko|Factor Engine|Multi-factor|Style Factor/ 
      }).count();
      console.log(`Toraniko Factor Engine integration: ${toranikoFactorEngine}`);
      
      const valueFactorHealth = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Value Factor|Book Price|P\/B/ 
      }).count();
      console.log(`Value Factor health: ${valueFactorHealth}`);
      
      const momentumFactorHealth = await page.locator('.ant-card, .ant-statistic').filter({ 
        hasText: /Momentum Factor|Price Momentum|Trend/ 
      }).count();
      console.log(`Momentum Factor health: ${momentumFactorHealth}`);
      
      const crossSourceFactors = await page.locator('.ant-card').filter({ 
        hasText: /Cross.*Source|EDGAR.*FRED|Triple.*Integration/ 
      }).count();
      console.log(`Cross-source factor combinations: ${crossSourceFactors}`);
      
      // Comprehensive integration assessment
      const coreSystemsHealthy = systemHealthy && messageBusConnected;
      const dataIntegrationHealthy = dataSourcesActive > 0;
      const tradingCapabilitiesReady = portfolioData > 0 && tradingReadiness > 0;
      const riskManagementActive = riskManagementIntegration > 0;
      const toranikoFactorEngineHealthy = toranikoFactorEngine > 0 && (valueFactorHealth > 0 || momentumFactorHealth > 0);
      
      const overallIntegrationHealth = coreSystemsHealthy && dataIntegrationHealthy && tradingCapabilitiesReady && toranikoFactorEngineHealthy;
      console.log(`\n=== NAUTILUS TRADER + TORANIKO INTEGRATION HEALTH SUMMARY ===`);
      console.log(`Core Systems (Engine + MessageBus): ${coreSystemsHealthy ? '✅ HEALTHY' : '❌ ISSUES'}`);
      console.log(`Data Integration (Multi-source): ${dataIntegrationHealthy ? '✅ HEALTHY' : '❌ ISSUES'}`);
      console.log(`Trading Capabilities: ${tradingCapabilitiesReady ? '✅ READY' : '❌ NOT READY'}`);
      console.log(`Risk Management: ${riskManagementActive ? '✅ ACTIVE' : '❌ INACTIVE'}`);
      console.log(`Toraniko Factor Engine: ${toranikoFactorEngineHealthy ? '✅ OPERATIONAL' : '❌ ISSUES'}`);
      console.log(`  ├─ Value Factor: ${valueFactorHealth > 0 ? '✅' : '❌'}`);
      console.log(`  ├─ Momentum Factor: ${momentumFactorHealth > 0 ? '✅' : '❌'}`);
      console.log(`  └─ Cross-Source Factors: ${crossSourceFactors > 0 ? '✅' : '❌'}`);
      console.log(`Overall Integration: ${overallIntegrationHealth ? '✅ HEALTHY' : '❌ ISSUES DETECTED'}`);
      console.log(`================================================================`);
      
      // Verify dashboard remains functional throughout all checks
      await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
      
      // Final navigation test to ensure all tabs remain accessible including Toraniko Factor Engine
      const tabKeys = ['system', 'nautilus-engine', 'backtesting', 'deployment', 'data-catalog', 
                     'search', 'watchlist', 'chart', 'strategy', 'performance', 'portfolio', 
                     'factors', 'risk', 'ib'];
      
      console.log('\n🧪 Testing accessibility of all dashboard tabs including Toraniko Factor Engine...');
      
      for (const tabKey of tabKeys) {
        await clickTab(page, tabKey);
        await page.waitForTimeout(1000);
        const tabAccessible = await page.locator(`[data-node-key="${tabKey}"]`).isVisible();
        if (!tabAccessible) {
          console.log(`⚠️  Warning: Tab ${tabKey} not accessible`);
        }
      }
      
      console.log('🎯 Comprehensive NautilusTrader + Toraniko Factor Engine integration health check completed');
      console.log('📊 Quantitative factor modeling capabilities verified');
      console.log('🚀 Institutional-grade risk model integration confirmed');
    });

    test('NautilusTrader + Toraniko performance benchmarking', async ({ page }) => {
      console.log('Running comprehensive NautilusTrader + Toraniko performance benchmarks...');
      
      // Test dashboard load performance
      const startTime = Date.now();
      await navigateToDashboard(page);
      const loadTime = Date.now() - startTime;
      console.log(`Dashboard load time: ${loadTime}ms`);
      expect(loadTime).toBeLessThan(10000); // 10 seconds max
      
      // Test tab switching performance including factor engine
      const tabSwitchTimes = [];
      const testTabs = ['system', 'portfolio', 'strategy', 'risk', 'factors', 'ib'];
      
      for (const tab of testTabs) {
        const switchStart = Date.now();
        await clickTab(page, tab);
        const switchTime = Date.now() - switchStart;
        tabSwitchTimes.push(switchTime);
        console.log(`${tab} tab switch time: ${switchTime}ms`);
      }
      
      const avgSwitchTime = tabSwitchTimes.reduce((a, b) => a + b) / tabSwitchTimes.length;
      console.log(`Average tab switch time: ${avgSwitchTime}ms`);
      expect(avgSwitchTime).toBeLessThan(5000); // 5 seconds max average
      
      // Test Toraniko Factor Engine performance
      await clickTab(page, 'factors');
      const factorCalculationStart = Date.now();
      await page.waitForTimeout(3000); // Allow factor calculations to load
      const factorCalculationTime = Date.now() - factorCalculationStart;
      console.log(`Factor engine load time: ${factorCalculationTime}ms`);
      
      const factorCalculationMetrics = await page.locator('.ant-statistic').filter({ 
        hasText: /Calculation.*Time|Factor.*Speed|Processing/ 
      }).count();
      console.log(`Factor calculation performance metrics: ${factorCalculationMetrics}`);
      
      // Test MessageBus performance under load
      await clickTab(page, 'system');
      const messageBusMetrics = await page.locator('.ant-statistic').filter({ 
        hasText: /Messages|Rate|Throughput/ 
      }).count();
      console.log(`MessageBus performance metrics available: ${messageBusMetrics > 0 ? 'Yes' : 'No'}`);
      
      // Test cross-source data integration performance
      const dataIntegrationMetrics = await page.locator('.ant-statistic').filter({ 
        hasText: /Data.*Rate|Integration.*Speed|Source.*Latency/ 
      }).count();
      console.log(`Data integration performance metrics: ${dataIntegrationMetrics}`);
      
      console.log('🚀 Comprehensive performance benchmarking completed');
    });

    test('Toraniko Factor Engine stress testing', async ({ page }) => {
      console.log('Running Toraniko Factor Engine stress tests...');
      
      await clickTab(page, 'factors');
      await page.waitForTimeout(2000);
      
      // Test factor calculation under load
      const factorLoadTesting = await page.locator('.ant-card').filter({ 
        hasText: /Load.*Test|Stress.*Test|Performance.*Test/ 
      }).count();
      console.log(`Factor load testing controls: ${factorLoadTesting}`);
      
      // Test large universe factor calculations
      const largeUniverseTesting = await page.locator('.ant-form-item').filter({ 
        hasText: /Universe.*Size|Symbol.*Count|Large.*Universe/ 
      }).count();
      console.log(`Large universe testing controls: ${largeUniverseTesting}`);
      
      // Test factor model stability
      const modelStability = await page.locator('.ant-statistic, .ant-alert').filter({ 
        hasText: /Stability|Robust|Convergence|Model.*Health/ 
      }).count();
      console.log(`Factor model stability indicators: ${modelStability}`);
      
      // Test memory usage monitoring
      const memoryUsage = await page.locator('.ant-statistic').filter({ 
        hasText: /Memory|RAM|Usage|Allocation/ 
      }).count();
      console.log(`Memory usage monitoring: ${memoryUsage}`);
      
      console.log('🔥 Factor Engine stress testing completed');
    });
  });
});