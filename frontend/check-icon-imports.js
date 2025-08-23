import fs from 'fs';
import path from 'path';

// List of potentially problematic icons based on our errors
const problematicIcons = [
  'ExclamationTriangleOutlined',
  'TrendingUpOutlined', 
  'TrendingDownOutlined',
  'ShieldOutlined'
];

const validIcons = [
  'RiseOutlined',
  'FallOutlined', 
  'WarningOutlined',
  'SafetyCertificateOutlined',
  'DashboardOutlined'
];

function findIconUsage(dir, icons) {
  const results = [];
  
  function searchFiles(directory) {
    const files = fs.readdirSync(directory);
    
    for (const file of files) {
      const fullPath = path.join(directory, file);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        if (!file.startsWith('.') && file !== 'node_modules') {
          searchFiles(fullPath);
        }
      } else if (file.endsWith('.tsx') || file.endsWith('.ts')) {
        const content = fs.readFileSync(fullPath, 'utf8');
        
        for (const icon of icons) {
          if (content.includes(icon)) {
            results.push({
              file: fullPath,
              icon: icon,
              lines: content.split('\n')
                .map((line, index) => ({ line: line.trim(), number: index + 1 }))
                .filter(({line}) => line.includes(icon))
                .map(({line, number}) => `${number}: ${line}`)
            });
          }
        }
      }
    }
  }
  
  searchFiles(dir);
  return results;
}

console.log('🔍 Searching for problematic icon imports...');

const srcDir = '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/src';
const problemResults = findIconUsage(srcDir, problematicIcons);

if (problemResults.length === 0) {
  console.log('✅ No problematic icons found!');
} else {
  console.log('❌ Found problematic icons:');
  problemResults.forEach(result => {
    console.log(`\n📄 File: ${result.file.replace(srcDir, '')}`);
    console.log(`🎯 Icon: ${result.icon}`);
    console.log('📝 Usage:');
    result.lines.forEach(line => console.log(`   ${line}`));
  });
}

// Generate replacement suggestions
console.log('\n💡 Suggested replacements:');
console.log('ExclamationTriangleOutlined → WarningOutlined');
console.log('TrendingUpOutlined → RiseOutlined'); 
console.log('TrendingDownOutlined → FallOutlined');
console.log('ShieldOutlined → SafetyCertificateOutlined');