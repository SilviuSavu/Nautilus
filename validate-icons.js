#!/usr/bin/env node
/**
 * Icon Validation Script - Add to CI/CD pipeline
 * Prevents deployment with missing Ant Design icons
 */

const fs = require('fs');
const path = require('path');

function validateIcons() {
  let availableIcons = [];
  try {
    const icons = require('./frontend/node_modules/@ant-design/icons');
    availableIcons = Object.keys(icons);
  } catch (error) {
    console.error('❌ Failed to load @ant-design/icons:', error.message);
    return false;
  }

  // Extract icons from code
  function findTsxFiles(dir, filesList = []) {
    const files = fs.readdirSync(dir);
    files.forEach(file => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory() && !['node_modules', '.git', 'dist', 'build', '.next'].includes(file)) {
        findTsxFiles(filePath, filesList);
      } else if (file.endsWith('.tsx') || file.endsWith('.ts')) {
        filesList.push(filePath);
      }
    });
    return filesList;
  }

  const tsxFiles = findTsxFiles('./frontend/src');
  const usedIcons = new Set();
  const missingIcons = new Set();
  
  tsxFiles.forEach(file => {
    try {
      const content = fs.readFileSync(file, 'utf8');
      const iconPattern = /([A-Z][a-zA-Z]*(?:Outlined|Filled|TwoTone))/g;
      const matches = content.match(iconPattern) || [];
      
      matches.forEach(icon => {
        usedIcons.add(icon);
        if (!availableIcons.includes(icon)) {
          missingIcons.add(icon);
        }
      });
    } catch (error) {
      // Skip files that can't be read
    }
  });
  
  if (missingIcons.size > 0) {
    console.error('❌ ICON VALIDATION FAILED');
    console.error('Missing icons found:', [...missingIcons].join(', '));
    console.error('Run "node fix-missing-icons.js" to fix automatically');
    return false;
  }
  
  console.log('✅ ICON VALIDATION PASSED');
  console.log(`Validated ${usedIcons.size} icons across ${tsxFiles.length} files`);
  return true;
}

// Run validation
const success = validateIcons();
process.exit(success ? 0 : 1);