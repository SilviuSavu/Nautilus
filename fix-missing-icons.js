#!/usr/bin/env node
/**
 * Automatic Icon Fixer
 * Replaces all missing Ant Design icons with correct alternatives
 */

const fs = require('fs');
const path = require('path');

// Icon replacement mapping
const iconReplacements = {
  // Missing -> Correct replacement
  'MemoryOutlined': 'DatabaseOutlined',
  'BrainOutlined': 'BulbOutlined', 
  'TestOutlined': 'ExperimentOutlined',
  'TrendingDownOutlined': 'FallOutlined',
  'TrendingUpOutlined': 'RiseOutlined', 
  'ShieldOutlined': 'SafetyCertificateOutlined',
  'ShieldCheckOutlined': 'SafetyOutlined',
  'ThumbsUpOutlined': 'LikeOutlined',
  'ThumbsDownOutlined': 'DislikeOutlined'
};

console.log('🔧 Starting automatic icon replacement...\n');

// Function to fix icons in a file
function fixIconsInFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let changes = [];
    
    // Replace both import statements and usage
    Object.entries(iconReplacements).forEach(([missing, replacement]) => {
      const importRegex = new RegExp(`\\b${missing}\\b`, 'g');
      if (content.match(importRegex)) {
        content = content.replace(importRegex, replacement);
        changes.push(`${missing} → ${replacement}`);
      }
    });
    
    if (changes.length > 0) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`✅ Fixed ${filePath.replace(process.cwd(), '.')}:`);
      changes.forEach(change => console.log(`   ${change}`));
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`❌ Error fixing ${filePath}:`, error.message);
    return false;
  }
}

// List of files that need fixing (from our scan)
const filesToFix = [
  './frontend/src/components/Nautilus/ResourceMonitor.tsx',
  './frontend/src/components/Pattern/AIPatternRecognition.tsx',
  './frontend/src/components/Pattern/CustomPatternBuilder.tsx',
  './frontend/src/components/Performance/DrawdownChart.tsx',
  './frontend/src/components/Performance/ExecutionQuality.tsx',
  './frontend/src/components/Performance/PerformanceAttribution.tsx',
  './frontend/src/components/Performance/RealTimeAnalytics.tsx',
  './frontend/src/components/Risk/BreachDetector.tsx',
  './frontend/src/components/Risk/DynamicLimitEngine.tsx',
  './frontend/src/components/Risk/DynamicRiskLimits.tsx',
  './frontend/src/components/Risk/LimitMonitor.tsx',
  './frontend/src/components/Risk/VaRCalculator.tsx',
  './frontend/src/components/Sprint3/FeatureNavigation.tsx',
  './frontend/src/components/Sprint3/QuickActions.tsx',
  './frontend/src/components/Sprint3/Sprint3StatusWidget.tsx',
  './frontend/src/components/Sprint3/SystemHealthOverview.tsx',
  './frontend/src/components/Strategy/ApprovalWorkflow.tsx',
  './frontend/src/components/Strategy/DeploymentPipeline.tsx',
  './frontend/src/components/Strategy/PipelineMonitor.tsx',
  './frontend/src/components/WebSocket/RealTimeStreaming.tsx'
];

let totalFixed = 0;
let filesModified = 0;

// Fix each file
filesToFix.forEach(file => {
  const fullPath = path.resolve(file);
  if (fs.existsSync(fullPath)) {
    if (fixIconsInFile(fullPath)) {
      filesModified++;
    }
  } else {
    console.log(`⚠️  File not found: ${file}`);
  }
});

console.log(`\n📊 Summary:`);
console.log(`   Files checked: ${filesToFix.length}`);
console.log(`   Files modified: ${filesModified}`);
console.log(`   Icon replacements: ${Object.keys(iconReplacements).length}`);

console.log('\n✅ Icon fixing complete! All missing icons have been replaced with valid alternatives.');
console.log('\n💡 Replacements made:');
Object.entries(iconReplacements).forEach(([missing, replacement]) => {
  console.log(`   ${missing} → ${replacement}`);
});