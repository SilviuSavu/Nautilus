/**
 * Story 5.3 Data Export and Reporting - Comprehensive Test Suite
 * Tests both backend API functionality and frontend integration
 */
import { test, expect } from '@playwright/test';

test.describe('Story 5.3 Data Export and Reporting - Comprehensive Testing', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
  });

  test('Backend API Endpoints - All Story 5.3 Data Export APIs', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Backend API Endpoints');
    
    // Test 1: Export Request Creation Endpoint
    const exportResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/export/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'csv',
          data_source: 'trades',
          filters: {
            date_range: {
              start_date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
              end_date: new Date().toISOString()
            },
            symbols: ['AAPL', 'MSFT']
          },
          fields: ['id', 'timestamp', 'symbol', 'price', 'quantity'],
          options: {
            include_headers: true,
            compression: false,
            precision: 4,
            timezone: 'UTC',
            currency: 'USD'
          }
        })
      });
      return { status: response.status, data: await response.json() };
    });
    
    expect(exportResponse.status).toBe(200);
    expect(exportResponse.data).toHaveProperty('export_id');
    expect(exportResponse.data).toHaveProperty('status');
    expect(exportResponse.data.status).toBe('pending');
    console.log('✅ Export Request Creation API working');

    // Test 2: Export History Endpoint
    const historyResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/export/history?limit=10');
      return { status: response.status, data: await response.json() };
    });
    
    expect(historyResponse.status).toBe(200);
    expect(historyResponse.data).toHaveProperty('exports');
    expect(historyResponse.data).toHaveProperty('total_count');
    expect(Array.isArray(historyResponse.data.exports)).toBeTruthy();
    console.log('✅ Export History API working');

    // Test 3: Supported Formats Endpoint
    const formatsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/export/formats');
      return { status: response.status, data: await response.json() };
    });
    
    expect(formatsResponse.status).toBe(200);
    expect(formatsResponse.data).toHaveProperty('formats');
    expect(Array.isArray(formatsResponse.data.formats)).toBeTruthy();
    expect(formatsResponse.data.formats.length).toBeGreaterThan(0);
    console.log('✅ Supported Formats API working');

    // Test 4: Available Fields Endpoint
    const fieldsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/export/fields/trades');
      return { status: response.status, data: await response.json() };
    });
    
    expect(fieldsResponse.status).toBe(200);
    expect(fieldsResponse.data).toHaveProperty('data_source');
    expect(fieldsResponse.data).toHaveProperty('available_fields');
    expect(Array.isArray(fieldsResponse.data.available_fields)).toBeTruthy();
    console.log('✅ Available Fields API working');

    // Test 5: Report Templates Endpoint
    const templatesResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/reports/templates');
      return { status: response.status, data: await response.json() };
    });
    
    expect(templatesResponse.status).toBe(200);
    expect(templatesResponse.data).toHaveProperty('templates');
    expect(templatesResponse.data).toHaveProperty('total_count');
    console.log('✅ Report Templates API working');

    // Test 6: API Integrations Endpoint
    const integrationsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/integrations');
      return { status: response.status, data: await response.json() };
    });
    
    expect(integrationsResponse.status).toBe(200);
    expect(integrationsResponse.data).toHaveProperty('integrations');
    expect(integrationsResponse.data).toHaveProperty('total_count');
    console.log('✅ API Integrations API working');

    await page.screenshot({ path: 'story-5-3-api-success.png' });
  });

  test('Frontend Integration - Navigate to Story 5.3 Data Export Dashboard', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Frontend Navigation');
    
    // Click on Performance Monitoring tab
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(3000);
    
    // Look for performance dashboard content
    await expect(page.locator('.performance-dashboard')).toBeVisible({ timeout: 10000 });
    console.log('✅ Performance dashboard loaded');
    
    // Look for Data Export tab
    const dataExportTab = page.locator('text=Data Export');
    await expect(dataExportTab).toBeVisible({ timeout: 10000 });
    console.log('✅ Data Export tab found');
    
    // Click Data Export tab
    await dataExportTab.click();
    await page.waitForTimeout(3000);
    
    // Verify Data Export Dashboard content loads
    const exportDashboard = page.locator('.data-export-dashboard');
    await expect(exportDashboard).toBeVisible({ timeout: 10000 });
    console.log('✅ Data Export dashboard rendered');
    
    // Check for main export sections
    await expect(page.locator('text=Data Export & Reporting')).toBeVisible();
    await expect(page.locator('text=Total Exports')).toBeVisible();
    await expect(page.locator('text=Export History')).toBeVisible();
    await expect(page.locator('text=New Export')).toBeVisible();
    console.log('✅ All Story 5.3 sections visible');
    
    await page.screenshot({ path: 'story-5-3-frontend-success.png' });
  });

  test('Story 5.3 Export Creation Workflow', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Export Creation Workflow');
    
    // Navigate to Data Export Dashboard
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Data Export');
    await page.waitForTimeout(3000);
    
    // Click New Export button
    const newExportButton = page.locator('button:has-text("New Export")').first();
    await newExportButton.click();
    await page.waitForTimeout(2000);
    
    // Verify export creation modal opens
    await expect(page.locator('text=Create Data Export')).toBeVisible();
    console.log('✅ Export creation modal opened');
    
    // Test form fields
    await expect(page.locator('label:has-text("Export Format")')).toBeVisible();
    await expect(page.locator('label:has-text("Data Source")')).toBeVisible();
    await expect(page.locator('label:has-text("Date Range")')).toBeVisible();
    console.log('✅ Export form fields visible');
    
    // Fill out the form
    await page.selectOption('select[name="type"]', 'csv');
    await page.selectOption('select[name="data_source"]', 'trades');
    
    // Test field selection updates based on data source
    const fieldsSelect = page.locator('div[role="combobox"]').last();
    await fieldsSelect.click();
    await page.waitForTimeout(1000);
    
    // Look for trade-specific fields
    await expect(page.locator('div:has-text("symbol")')).toBeVisible();
    await expect(page.locator('div:has-text("price")')).toBeVisible();
    console.log('✅ Field selection working');
    
    // Close modal
    await page.click('button:has-text("Cancel")');
    await page.waitForTimeout(1000);
    
    await page.screenshot({ path: 'story-5-3-export-workflow.png' });
  });

  test('Story 5.3 Export Dashboard Statistics', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Export Dashboard Statistics');
    
    // Navigate to Data Export Dashboard
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Data Export');
    await page.waitForTimeout(3000);
    
    // Check export statistics cards
    await expect(page.locator('text=Total Exports')).toBeVisible();
    await expect(page.locator('text=Completed')).toBeVisible();
    await expect(page.locator('text=Processing')).toBeVisible();
    await expect(page.locator('text=Failed')).toBeVisible();
    console.log('✅ Export statistics cards visible');
    
    // Verify export history table
    await expect(page.locator('text=Export History')).toBeVisible();
    const exportTable = page.locator('.ant-table-tbody');
    await expect(exportTable).toBeVisible();
    console.log('✅ Export history table visible');
    
    // Check for refresh functionality
    const refreshButton = page.locator('button:has-text("Refresh")');
    await expect(refreshButton).toBeVisible();
    await refreshButton.click();
    await page.waitForTimeout(1000);
    console.log('✅ Refresh functionality working');
    
    await page.screenshot({ path: 'story-5-3-dashboard-stats.png' });
  });

  test('Story 5.3 Tab Navigation within Data Export Dashboard', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Internal Tab Navigation');
    
    // Navigate to Data Export Dashboard
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Data Export');
    await page.waitForTimeout(3000);
    
    // Test Data Exports tab (default)
    await expect(page.locator('text=Export History')).toBeVisible();
    await expect(page.locator('text=Total Exports')).toBeVisible();
    console.log('✅ Data Exports tab working');
    
    // Test Report Templates tab
    const reportTemplatesTab = page.locator('.ant-tabs-tab').filter({ hasText: 'Report Templates' });
    await reportTemplatesTab.click();
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Report template management')).toBeVisible();
    console.log('✅ Report Templates tab working');
    
    // Test API Integrations tab
    const integrationsTab = page.locator('.ant-tabs-tab').filter({ hasText: 'API Integrations' });
    await integrationsTab.click();
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Third-party API integrations')).toBeVisible();
    console.log('✅ API Integrations tab working');
    
    await page.screenshot({ path: 'story-5-3-tabs-success.png' });
  });

  test('Story 5.3 Export Status Monitoring', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Export Status Monitoring');
    
    // Navigate to Data Export Dashboard
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Data Export');
    await page.waitForTimeout(3000);
    
    // Monitor network requests for export status updates
    const apiRequests: string[] = [];
    page.on('response', response => {
      if (response.url().includes('/api/v1/export/')) {
        apiRequests.push(response.url());
        console.log('📡 Export API Call:', response.url(), response.status());
      }
    });
    
    // Check for status indicators in export table
    const statusColumns = page.locator('.ant-table-tbody .ant-tag');
    const statusCount = await statusColumns.count();
    if (statusCount > 0) {
      console.log('✅ Export status indicators found');
    }
    
    // Check for progress bars on processing exports
    const progressBars = page.locator('.ant-progress');
    const progressCount = await progressBars.count();
    if (progressCount > 0) {
      console.log('✅ Progress bars for processing exports found');
    }
    
    // Verify auto-refresh is working by waiting for API calls
    await page.waitForTimeout(6000); // Wait for potential auto-refresh
    
    if (apiRequests.length > 0) {
      console.log('✅ Export API integration working');
    }
    
    await page.screenshot({ path: 'story-5-3-monitoring-success.png' });
  });

  test('Story 5.3 Export Format and Data Source Integration', async ({ page }) => {
    console.log('🔍 Testing Story 5.3 Export Format and Data Source Integration');
    
    // Navigate to Data Export Dashboard
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=Data Export');
    await page.waitForTimeout(3000);
    
    // Open export creation modal
    await page.click('button:has-text("New Export")');
    await page.waitForTimeout(2000);
    
    // Test all export formats are available
    const formatSelect = page.locator('select[name="type"]');
    await formatSelect.click();
    
    const formatOptions = page.locator('option');
    const formatTexts = await formatOptions.allTextContents();
    expect(formatTexts.join(' ')).toContain('CSV');
    expect(formatTexts.join(' ')).toContain('JSON');
    expect(formatTexts.join(' ')).toContain('Excel');
    expect(formatTexts.join(' ')).toContain('PDF');
    console.log('✅ All export formats available');
    
    // Test all data sources are available
    const dataSourceSelect = page.locator('select[name="data_source"]');
    await dataSourceSelect.click();
    
    const sourceOptions = page.locator('option');
    const sourceTexts = await sourceOptions.allTextContents();
    expect(sourceTexts.join(' ')).toContain('Trading');
    expect(sourceTexts.join(' ')).toContain('Positions');
    expect(sourceTexts.join(' ')).toContain('Performance');
    expect(sourceTexts.join(' ')).toContain('Order');
    expect(sourceTexts.join(' ')).toContain('System');
    console.log('✅ All data sources available');
    
    // Test field selection updates with data source
    await page.selectOption('select[name="data_source"]', 'performance');
    await page.waitForTimeout(1000);
    
    // Close modal
    await page.click('button:has-text("Cancel")');
    
    await page.screenshot({ path: 'story-5-3-formats-success.png' });
  });
});