// Debug script to check Performance API endpoints
const BASE_URL = 'http://localhost:8000';

async function checkEndpoint(endpoint, description) {
  try {
    console.log(`\n🔍 Testing: ${description}`);
    console.log(`📡 URL: ${BASE_URL}${endpoint}`);
    
    const response = await fetch(`${BASE_URL}${endpoint}`);
    console.log(`✅ Status: ${response.status} ${response.statusText}`);
    
    if (response.ok) {
      const data = await response.json();
      console.log(`📊 Data received:`, JSON.stringify(data, null, 2).substring(0, 200) + '...');
    } else {
      const errorText = await response.text();
      console.log(`❌ Error response:`, errorText);
    }
  } catch (error) {
    console.log(`🚨 Network error:`, error.message);
  }
}

async function debugPerformanceAPI() {
  console.log('🐛 Debugging Performance Tab API Endpoints\n');
  
  // Check basic health
  await checkEndpoint('/health', 'Backend Health Check');
  
  // Check strategies endpoints
  await checkEndpoint('/api/v1/strategies/active', 'Active Strategies');
  await checkEndpoint('/api/v1/performance/aggregate', 'Performance Aggregate');
  await checkEndpoint('/api/v1/performance/history?strategy_id=all&start_date=2024-07-21T00:00:00.000Z&end_date=2024-08-21T00:00:00.000Z', 'Performance History');
  
  console.log('\n✅ Performance API debugging complete');
}

// Run the debug
debugPerformanceAPI().catch(console.error);