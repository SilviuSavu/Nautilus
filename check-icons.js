#!/usr/bin/env node
/**
 * Comprehensive Ant Design Icons Checker
 * Scans the codebase and verifies all used icons exist
 */

const fs = require('fs');
const path = require('path');

// Get all icons from @ant-design/icons
let availableIcons = [];
try {
  const icons = require('./frontend/node_modules/@ant-design/icons');
  availableIcons = Object.keys(icons).filter(key => key.endsWith('Outlined') || key.endsWith('Filled') || key.endsWith('TwoTone'));
  console.log(`✅ Found ${availableIcons.length} available Ant Design icons`);
} catch (error) {
  console.error('❌ Failed to load @ant-design/icons:', error.message);
  process.exit(1);
}

// Function to recursively find all TypeScript/JSX files
function findTsxFiles(dir, filesList = []) {
  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      // Skip node_modules and other irrelevant directories
      if (!['node_modules', '.git', 'dist', 'build', '.next'].includes(file)) {
        findTsxFiles(filePath, filesList);
      }
    } else if (file.endsWith('.tsx') || file.endsWith('.ts')) {
      filesList.push(filePath);
    }
  });
  return filesList;
}

// Function to extract icon names from file content
function extractIconNames(content) {
  const iconPattern = /([A-Z][a-zA-Z]*(?:Outlined|Filled|TwoTone))/g;
  const matches = content.match(iconPattern) || [];
  return [...new Set(matches)]; // Remove duplicates
}

// Main function
function checkIcons() {
  console.log('\n🔍 Scanning codebase for icon usage...\n');
  
  const frontendDir = './frontend/src';
  const tsxFiles = findTsxFiles(frontendDir);
  
  const usedIcons = new Set();
  const missingIcons = new Set();
  const fileIconMap = new Map();
  
  tsxFiles.forEach(file => {
    try {
      const content = fs.readFileSync(file, 'utf8');
      const iconsInFile = extractIconNames(content);
      
      if (iconsInFile.length > 0) {
        fileIconMap.set(file, iconsInFile);
        iconsInFile.forEach(icon => {
          usedIcons.add(icon);
          if (!availableIcons.includes(icon)) {
            missingIcons.add(icon);
          }
        });
      }
    } catch (error) {
      console.error(`❌ Error reading file ${file}:`, error.message);
    }
  });
  
  console.log(`📊 Statistics:`);
  console.log(`   Total .tsx/.ts files scanned: ${tsxFiles.length}`);
  console.log(`   Files using icons: ${fileIconMap.size}`);
  console.log(`   Unique icons used: ${usedIcons.size}`);
  console.log(`   Missing icons: ${missingIcons.size}\n`);
  
  if (missingIcons.size > 0) {
    console.log('❌ Missing Icons:');
    missingIcons.forEach(icon => {
      console.log(`   - ${icon}`);
      
      // Suggest alternatives
      const suggestions = availableIcons.filter(available => {
        const iconName = icon.replace(/(?:Outlined|Filled|TwoTone)$/, '');
        const availableName = available.replace(/(?:Outlined|Filled|TwoTone)$/, '');
        return availableName.toLowerCase().includes(iconName.toLowerCase()) ||
               iconName.toLowerCase().includes(availableName.toLowerCase());
      });
      
      if (suggestions.length > 0) {
        console.log(`     Suggestions: ${suggestions.slice(0, 3).join(', ')}`);
      }
    });
    
    console.log('\n📁 Files with missing icons:');
    fileIconMap.forEach((icons, file) => {
      const missing = icons.filter(icon => missingIcons.has(icon));
      if (missing.length > 0) {
        console.log(`   ${file}:`);
        missing.forEach(icon => {
          console.log(`     - ${icon}`);
        });
      }
    });
    
    console.log('\n💡 To fix missing icons, you can:');
    console.log('   1. Replace with suggested alternatives');
    console.log('   2. Use generic icons like WarningOutlined, InfoCircleOutlined');
    console.log('   3. Check Ant Design documentation for correct icon names');
    
  } else {
    console.log('✅ All icons are available! No missing icons found.');
  }
  
  console.log('\n🔧 Available icon types:');
  const iconTypes = {
    outlined: availableIcons.filter(icon => icon.endsWith('Outlined')).length,
    filled: availableIcons.filter(icon => icon.endsWith('Filled')).length,
    twoTone: availableIcons.filter(icon => icon.endsWith('TwoTone')).length
  };
  console.log(`   Outlined: ${iconTypes.outlined}`);
  console.log(`   Filled: ${iconTypes.filled}`);
  console.log(`   TwoTone: ${iconTypes.twoTone}`);
  
  return missingIcons.size === 0;
}

// Run the check
const success = checkIcons();
process.exit(success ? 0 : 1);