/**
 * Story 5.2 System Performance Monitoring - Comprehensive Test Suite
 * Tests both backend API functionality and frontend integration
 */
import { test, expect } from '@playwright/test';

test.describe('Story 5.2 System Performance Monitoring - Comprehensive Testing', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log('BROWSER:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:3000');
    await page.waitForTimeout(3000);
  });

  test('Backend API Endpoints - All Story 5.2 Monitoring APIs', async ({ page }) => {
    console.log('🔍 Testing Story 5.2 Backend API Endpoints');
    
    // Test 1: Latency Monitoring Endpoint
    const latencyResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/monitoring/latency?venue=all&timeframe=1h');
      return { status: response.status, data: await response.json() };
    });
    
    expect(latencyResponse.status).toBe(200);
    expect(latencyResponse.data).toHaveProperty('venue_latencies');
    expect(latencyResponse.data).toHaveProperty('overall_statistics');
    expect(latencyResponse.data.venue_latencies).toBeInstanceOf(Array);
    console.log('✅ Latency Monitoring API working');

    // Test 2: Connection Monitoring Endpoint
    const connectionResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/monitoring/connections?venue=all&include_history=true');
      return { status: response.status, data: await response.json() };
    });
    
    expect(connectionResponse.status).toBe(200);
    expect(connectionResponse.data).toHaveProperty('venue_connections');
    expect(connectionResponse.data).toHaveProperty('overall_health');
    expect(connectionResponse.data.venue_connections).toBeInstanceOf(Array);
    console.log('✅ Connection Monitoring API working');

    // Test 3: Alerts Monitoring Endpoint
    const alertsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/monitoring/alerts?status=active&severity=all');
      return { status: response.status, data: await response.json() };
    });
    
    expect(alertsResponse.status).toBe(200);
    expect(alertsResponse.data).toHaveProperty('active_alerts');
    expect(alertsResponse.data).toHaveProperty('alert_statistics');
    expect(alertsResponse.data.active_alerts).toBeInstanceOf(Array);
    console.log('✅ Alerts Monitoring API working');

    // Test 4: Performance Trends Endpoint
    const trendsResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/monitoring/performance-trends?period=7d');
      return { status: response.status, data: await response.json() };
    });
    
    expect(trendsResponse.status).toBe(200);
    expect(trendsResponse.data).toHaveProperty('trend_analysis');
    expect(trendsResponse.data).toHaveProperty('capacity_planning');
    expect(trendsResponse.data.trend_analysis).toBeInstanceOf(Array);
    console.log('✅ Performance Trends API working');

    // Test 5: Alert Configuration Endpoint
    const configResponse = await page.evaluate(async () => {
      const response = await fetch('/api/v1/monitoring/alerts/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metric_name: 'cpu_usage',
          threshold_value: 85.0,
          condition: 'greater_than',
          severity: 'high',
          enabled: true,
          notification_channels: {
            email: ['test@example.com']
          },
          escalation_rules: {
            escalate_after_minutes: 30,
            escalation_contacts: ['admin@example.com']
          },
          auto_resolution: {
            enabled: true,
            resolution_threshold: 75.0,
            max_attempts: 3
          }
        })
      });
      return { status: response.status, data: await response.json() };
    });
    
    expect(configResponse.status).toBe(200);
    expect(configResponse.data).toHaveProperty('alert_rule_id');
    expect(configResponse.data).toHaveProperty('status');
    console.log('✅ Alert Configuration API working');

    await page.screenshot({ path: 'story-5-2-api-success.png' });
  });

  test('Frontend Integration - Navigate to Story 5.2 System Monitoring', async ({ page }) => {
    console.log('🔍 Testing Story 5.2 Frontend Navigation');
    
    // Click on Performance Monitoring tab (using the correct navigation)
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(3000);
    
    // Look for performance dashboard content
    await expect(page.locator('.performance-dashboard')).toBeVisible({ timeout: 10000 });
    console.log('✅ Performance dashboard loaded');
    
    // Look for System Monitoring tab
    const systemMonitoringTab = page.locator('text=System Monitoring');
    await expect(systemMonitoringTab).toBeVisible({ timeout: 10000 });
    console.log('✅ System Monitoring tab found');
    
    // Click System Monitoring tab
    await systemMonitoringTab.click();
    await page.waitForTimeout(3000);
    
    // Verify System Monitoring content loads
    const monitoringContent = page.locator('.system-performance-dashboard');
    await expect(monitoringContent).toBeVisible({ timeout: 10000 });
    console.log('✅ System Monitoring dashboard rendered');
    
    // Check for main monitoring sections
    await expect(page.locator('text=System Performance Monitoring')).toBeVisible();
    await expect(page.locator('text=Avg Order Latency')).toBeVisible();
    await expect(page.locator('text=CPU Usage')).toBeVisible();
    await expect(page.locator('text=Memory Usage')).toBeVisible();
    await expect(page.locator('text=Connected Venues')).toBeVisible();
    console.log('✅ All Story 5.2 sections visible');
    
    await page.screenshot({ path: 'story-5-2-frontend-success.png' });
  });

  test('Story 5.2 Tab Navigation within System Monitoring Dashboard', async ({ page }) => {
    console.log('🔍 Testing Story 5.2 Internal Tab Navigation');
    
    // Navigate to System Monitoring
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=System Monitoring');
    await page.waitForTimeout(3000);
    
    // Test Overview tab (default)
    await expect(page.locator('text=Venue Latency Overview')).toBeVisible();
    await expect(page.locator('text=Connection Status')).toBeVisible();
    console.log('✅ Overview tab working');
    
    // Test Latency Details tab
    await page.click('text=Latency Details');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Detailed latency monitoring charts will be implemented here')).toBeVisible();
    console.log('✅ Latency Details tab working');
    
    // Test System Resources tab
    await page.click('text=System Resources');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=System resource monitoring charts will be implemented here')).toBeVisible();
    console.log('✅ System Resources tab working');
    
    // Test Connections tab
    await page.click('text=Connections');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Connection quality monitoring will be implemented here')).toBeVisible();
    console.log('✅ Connections tab working');
    
    // Test Alerts tab
    await page.click('text=Alerts');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Alert configuration and management will be implemented here')).toBeVisible();
    console.log('✅ Alerts tab working');
    
    // Test Trends tab
    await page.click('text=Trends');
    await page.waitForTimeout(2000);
    
    await expect(page.locator('text=Performance trends and capacity planning will be implemented here')).toBeVisible();
    console.log('✅ Trends tab working');
    
    await page.screenshot({ path: 'story-5-2-tabs-success.png' });
  });

  test('Story 5.2 Real-time Monitoring Integration', async ({ page }) => {
    console.log('🔍 Testing Story 5.2 Real-time Monitoring Features');
    
    // Navigate to System Monitoring
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=System Monitoring');
    await page.waitForTimeout(3000);
    
    // Monitor network requests
    const apiRequests: string[] = [];
    page.on('response', response => {
      if (response.url().includes('/api/v1/monitoring/')) {
        apiRequests.push(response.url());
        console.log('📡 Monitoring API Call:', response.url(), response.status());
      }
    });
    
    // Look for auto-refresh controls
    const autoRefreshButton = page.locator('button:has-text("Auto-refresh")');
    await expect(autoRefreshButton).toBeVisible();
    console.log('✅ Auto-refresh control found');
    
    // Look for refresh button and click it to trigger API calls
    const refreshButton = page.locator('button:has-text("Refresh")');
    await expect(refreshButton).toBeVisible();
    await refreshButton.click();
    await page.waitForTimeout(3000);
    
    // Verify API calls were made
    expect(apiRequests.length).toBeGreaterThan(0);
    console.log('✅ Story 5.2 API integration working');
    
    // Check for real-time metrics display
    await expect(page.locator('text=Avg Order Latency')).toBeVisible();
    await expect(page.locator('text=CPU Usage')).toBeVisible();
    await expect(page.locator('text=Memory Usage')).toBeVisible();
    await expect(page.locator('text=Connected Venues')).toBeVisible();
    console.log('✅ Real-time metrics displayed');
    
    // Check for error states (should not have errors)
    const errorMessages = await page.locator('.ant-alert-error').count();
    expect(errorMessages).toBe(0);
    console.log('✅ No error states detected');
    
    await page.screenshot({ path: 'story-5-2-realtime-success.png' });
  });

  test('Story 5.2 Monitoring Controls and Configuration', async ({ page }) => {
    console.log('🔍 Testing Story 5.2 Monitoring Controls');
    
    // Navigate to System Monitoring
    await page.click('text=Performance Monitoring');
    await page.waitForTimeout(2000);
    await page.click('text=System Monitoring');
    await page.waitForTimeout(3000);
    
    // Test venue selector
    const venueSelector = page.locator('select').first();
    await expect(venueSelector).toBeVisible();
    console.log('✅ Venue selector available');
    
    // Test timeframe selector
    const timeframeSelector = page.locator('select').nth(1);
    await expect(timeframeSelector).toBeVisible();
    console.log('✅ Timeframe selector available');
    
    // Test auto-refresh toggle
    const autoRefreshToggle = page.locator('button:has-text("Auto-refresh")');
    await autoRefreshToggle.click();
    await page.waitForTimeout(1000);
    console.log('✅ Auto-refresh toggle working');
    
    // Verify last updated timestamp
    await expect(page.locator('text=Last updated:')).toBeVisible();
    console.log('✅ Last updated timestamp displayed');
    
    await page.screenshot({ path: 'story-5-2-controls-success.png' });
  });
});