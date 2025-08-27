#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

// Icon mapping from non-existent to existing icons
const iconReplacements = {
  'BrainOutlined': 'ExperimentOutlined',
  'ProcessorOutlined': 'ControlOutlined', 
  'TrendingUpOutlined': 'LineChartOutlined',
  'MemoryStickOutlined': 'DatabaseOutlined',
  'BranchesOutlined': 'GitlabOutlined',
  'HeatmapOutlined': 'DotChartOutlined',
  'DatasetOutlined': 'TableOutlined',
  'ChipOutlined': 'ApiOutlined',
  'GpuOutlined': 'CodepenOutlined',
  'NeuralNetworkOutlined': 'ShareAltOutlined'
};

function replaceIconsInFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let changed = false;
    
    for (const [oldIcon, newIcon] of Object.entries(iconReplacements)) {
      if (content.includes(oldIcon)) {
        content = content.replaceAll(oldIcon, newIcon);
        changed = true;
        console.log(`✅ Replaced ${oldIcon} with ${newIcon} in ${filePath}`);
      }
    }
    
    if (changed) {
      fs.writeFileSync(filePath, content, 'utf8');
      return true;
    }
    return false;
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
    return false;
  }
}

function findAndReplaceIcons(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  let totalFixed = 0;
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
      totalFixed += findAndReplaceIcons(fullPath);
    } else if (entry.isFile() && (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts'))) {
      if (replaceIconsInFile(fullPath)) {
        totalFixed++;
      }
    }
  }
  
  return totalFixed;
}

console.log('🔧 Fixing non-existent Ant Design icons...');
const srcPath = '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/frontend/src';
const filesFixed = findAndReplaceIcons(srcPath);
console.log(`✨ Fixed icons in ${filesFixed} files`);