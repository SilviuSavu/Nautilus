#!/usr/bin/env node
/**
 * Production Build Optimization Script
 * Comprehensive build optimization for Sprint 3 frontend
 */

import { execSync } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Configuration
const CONFIG = {
  buildDir: path.join(rootDir, 'dist'),
  srcDir: path.join(rootDir, 'src'),
  publicDir: path.join(rootDir, 'public'),
  reportDir: path.join(rootDir, 'reports'),
  
  // Bundle size thresholds (in KB)
  thresholds: {
    maxBundleSize: 1500,
    maxChunkSize: 500,
    maxAssetSize: 100
  },
  
  // Optimization flags
  enableMinification: true,
  enableCompression: true,
  enableTreeShaking: true,
  enableSplitting: true,
  enableSourceMaps: process.env.NODE_ENV !== 'production'
};

// Utilities
const log = (message, type = 'info') => {
  const colors = {
    info: '\x1b[36m',    // Cyan
    success: '\x1b[32m', // Green  
    warning: '\x1b[33m', // Yellow
    error: '\x1b[31m',   // Red
    reset: '\x1b[0m'     // Reset
  };
  
  console.log(`${colors[type]}[${type.toUpperCase()}]${colors.reset} ${message}`);
};

const exec = (command, options = {}) => {
  try {
    return execSync(command, { 
      encoding: 'utf8', 
      cwd: rootDir,
      stdio: 'inherit',
      ...options 
    });
  } catch (error) {
    log(`Command failed: ${command}`, 'error');
    log(error.message, 'error');
    process.exit(1);
  }
};

const execSilent = (command, options = {}) => {
  try {
    return execSync(command, { 
      encoding: 'utf8', 
      cwd: rootDir,
      stdio: 'pipe',
      ...options 
    });
  } catch (error) {
    return null;
  }
};

const formatBytes = (bytes) => {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i) * 10) / 10} ${sizes[i]}`;
};

// Main optimization functions
async function cleanBuildDirectory() {
  log('Cleaning build directory...');
  try {
    await fs.rm(CONFIG.buildDir, { recursive: true, force: true });
    await fs.rm(CONFIG.reportDir, { recursive: true, force: true });
    log('Build directory cleaned successfully', 'success');
  } catch (error) {
    log(`Failed to clean build directory: ${error.message}`, 'error');
  }
}

async function checkDependencies() {
  log('Checking dependencies...');
  
  const packageJsonPath = path.join(rootDir, 'package.json');
  const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf8'));
  
  // Check for unused dependencies
  const installedDeps = Object.keys({
    ...packageJson.dependencies,
    ...packageJson.devDependencies
  });
  
  log(`Found ${installedDeps.length} dependencies`, 'info');
  
  // Update dependencies if needed
  const outdated = execSilent('npm outdated --json');
  if (outdated) {
    const outdatedPackages = JSON.parse(outdated);
    const count = Object.keys(outdatedPackages).length;
    if (count > 0) {
      log(`${count} packages have updates available`, 'warning');
    }
  }
  
  log('Dependency check completed', 'success');
}

async function runTypeScriptCheck() {
  log('Running TypeScript type checking...');
  
  try {
    exec('npx tsc --noEmit --project tsconfig.build.json');
    log('TypeScript check passed', 'success');
  } catch (error) {
    log('TypeScript check failed', 'error');
    process.exit(1);
  }
}

async function runLinting() {
  log('Running ESLint...');
  
  try {
    exec('npx eslint src --ext .ts,.tsx --max-warnings 0');
    log('Linting passed', 'success');
  } catch (error) {
    log('Linting failed', 'error');
    process.exit(1);
  }
}

async function runTests() {
  log('Running tests...');
  
  try {
    exec('npm run test:coverage');
    log('Tests passed', 'success');
  } catch (error) {
    log('Tests failed', 'warning');
    // Don't exit on test failure in CI
  }
}

async function buildProduction() {
  log('Building for production...');
  
  const buildStart = Date.now();
  
  // Set production environment
  process.env.NODE_ENV = 'production';
  process.env.GENERATE_SOURCEMAP = CONFIG.enableSourceMaps ? 'true' : 'false';
  
  try {
    exec('npm run build');
    
    const buildTime = Date.now() - buildStart;
    log(`Production build completed in ${buildTime}ms`, 'success');
  } catch (error) {
    log('Production build failed', 'error');
    process.exit(1);
  }
}

async function analyzeBundleSize() {
  log('Analyzing bundle size...');
  
  try {
    // Create reports directory
    await fs.mkdir(CONFIG.reportDir, { recursive: true });
    
    // Get bundle analyzer report
    exec('npx vite-bundle-analyzer --mode production --out-dir reports');
    
    // Analyze build directory
    const distStats = await analyzeDirectory(CONFIG.buildDir);
    
    // Check against thresholds
    const warnings = [];
    
    if (distStats.totalSize > CONFIG.thresholds.maxBundleSize * 1024) {
      warnings.push(`Bundle size (${formatBytes(distStats.totalSize)}) exceeds threshold (${formatBytes(CONFIG.thresholds.maxBundleSize * 1024)})`);
    }
    
    distStats.files.forEach(file => {
      if (file.size > CONFIG.thresholds.maxAssetSize * 1024 && file.name.endsWith('.js')) {
        warnings.push(`Chunk ${file.name} (${formatBytes(file.size)}) exceeds threshold`);
      }
    });
    
    if (warnings.length > 0) {
      log('Bundle size warnings:', 'warning');
      warnings.forEach(warning => log(`  - ${warning}`, 'warning'));
    } else {
      log('Bundle size within thresholds', 'success');
    }
    
    log(`Total bundle size: ${formatBytes(distStats.totalSize)}`, 'info');
    log(`Number of chunks: ${distStats.files.filter(f => f.name.endsWith('.js')).length}`, 'info');
    
  } catch (error) {
    log(`Bundle analysis failed: ${error.message}`, 'warning');
  }
}

async function analyzeDirectory(dirPath) {
  const files = [];
  let totalSize = 0;
  
  async function scanDir(currentPath) {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);
      
      if (entry.isDirectory()) {
        await scanDir(fullPath);
      } else {
        const stats = await fs.stat(fullPath);
        const relativePath = path.relative(CONFIG.buildDir, fullPath);
        
        files.push({
          name: relativePath,
          size: stats.size,
          type: path.extname(entry.name)
        });
        
        totalSize += stats.size;
      }
    }
  }
  
  await scanDir(dirPath);
  
  return { files, totalSize };
}

async function optimizeAssets() {
  log('Optimizing assets...');
  
  try {
    // Optimize images if imagemin is available
    const hasImagemin = execSilent('npx imagemin --version');
    if (hasImagemin) {
      exec(`npx imagemin "${CONFIG.buildDir}/**/*.{jpg,png,svg}" --out-dir="${CONFIG.buildDir}"`);
      log('Images optimized', 'success');
    }
    
    // Compress JavaScript and CSS files
    if (CONFIG.enableCompression) {
      const gzipCmd = process.platform === 'win32' ? 'powershell' : 'gzip';
      if (gzipCmd === 'gzip') {
        exec(`find "${CONFIG.buildDir}" -type f \\( -name "*.js" -o -name "*.css" \\) -exec gzip -k {} \\;`);
        log('Assets compressed with gzip', 'success');
      }
    }
    
  } catch (error) {
    log(`Asset optimization failed: ${error.message}`, 'warning');
  }
}

async function generateServiceWorker() {
  log('Generating service worker...');
  
  try {
    // Check if workbox is available
    const hasWorkbox = execSilent('npx workbox --version');
    if (!hasWorkbox) {
      log('Workbox not available, skipping service worker generation', 'warning');
      return;
    }
    
    // Generate service worker
    const workboxConfig = {
      globDirectory: CONFIG.buildDir,
      globPatterns: [
        '**/*.{js,css,html,png,jpg,jpeg,svg,ico,woff,woff2}'
      ],
      swDest: path.join(CONFIG.buildDir, 'sw.js'),
      runtimeCaching: [
        {
          urlPattern: /^https:\/\/api\./,
          handler: 'NetworkFirst',
          options: {
            cacheName: 'api-cache',
            expiration: {
              maxEntries: 100,
              maxAgeSeconds: 300 // 5 minutes
            }
          }
        }
      ]
    };
    
    await fs.writeFile(
      path.join(rootDir, 'workbox-config.js'),
      `module.exports = ${JSON.stringify(workboxConfig, null, 2)};`
    );
    
    exec('npx workbox generateSW workbox-config.js');
    log('Service worker generated', 'success');
    
  } catch (error) {
    log(`Service worker generation failed: ${error.message}`, 'warning');
  }
}

async function runSecurityAudit() {
  log('Running security audit...');
  
  try {
    exec('npm audit --audit-level=moderate');
    log('Security audit passed', 'success');
  } catch (error) {
    log('Security audit found issues', 'warning');
    // Don't exit on audit failures in production build
  }
}

async function generateBuildReport() {
  log('Generating build report...');
  
  try {
    await fs.mkdir(CONFIG.reportDir, { recursive: true });
    
    const distStats = await analyzeDirectory(CONFIG.buildDir);
    const packageJson = JSON.parse(
      await fs.readFile(path.join(rootDir, 'package.json'), 'utf8')
    );
    
    const report = {
      timestamp: new Date().toISOString(),
      version: packageJson.version,
      buildInfo: {
        nodeVersion: process.version,
        npmVersion: execSilent('npm --version')?.trim(),
        buildTime: Date.now(),
        environment: process.env.NODE_ENV || 'development'
      },
      bundleStats: {
        totalSize: distStats.totalSize,
        totalSizeFormatted: formatBytes(distStats.totalSize),
        fileCount: distStats.files.length,
        chunks: distStats.files
          .filter(f => f.name.endsWith('.js'))
          .map(f => ({
            name: f.name,
            size: f.size,
            sizeFormatted: formatBytes(f.size)
          })),
        assets: distStats.files
          .filter(f => !f.name.endsWith('.js') && !f.name.endsWith('.css'))
          .map(f => ({
            name: f.name,
            size: f.size,
            sizeFormatted: formatBytes(f.size)
          }))
      },
      thresholds: CONFIG.thresholds,
      warnings: distStats.totalSize > CONFIG.thresholds.maxBundleSize * 1024 ? [
        `Bundle size exceeds threshold`
      ] : []
    };
    
    await fs.writeFile(
      path.join(CONFIG.reportDir, 'build-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    log('Build report generated', 'success');
    log(`Build report saved to: ${path.join(CONFIG.reportDir, 'build-report.json')}`, 'info');
    
  } catch (error) {
    log(`Build report generation failed: ${error.message}`, 'warning');
  }
}

async function runPerformanceTest() {
  log('Running performance tests...');
  
  try {
    // Run Lighthouse if available
    const hasLighthouse = execSilent('npx lighthouse --version');
    if (hasLighthouse) {
      exec('npx lighthouse --output=json --output-path=reports/lighthouse-report.json http://localhost:3000');
      log('Lighthouse performance test completed', 'success');
    } else {
      log('Lighthouse not available, skipping performance test', 'warning');
    }
  } catch (error) {
    log(`Performance test failed: ${error.message}`, 'warning');
  }
}

// Main execution function
async function main() {
  const startTime = Date.now();
  
  log('🚀 Starting Sprint 3 Frontend Production Build Optimization', 'info');
  log('═'.repeat(60), 'info');
  
  try {
    // Pre-build steps
    await cleanBuildDirectory();
    await checkDependencies();
    await runTypeScriptCheck();
    await runLinting();
    
    // Optional: Run tests
    if (process.argv.includes('--with-tests')) {
      await runTests();
    }
    
    // Build step
    await buildProduction();
    
    // Post-build optimizations
    await analyzeBundleSize();
    await optimizeAssets();
    await generateServiceWorker();
    
    // Security and reporting
    if (process.argv.includes('--audit')) {
      await runSecurityAudit();
    }
    
    await generateBuildReport();
    
    // Optional: Performance testing
    if (process.argv.includes('--performance')) {
      await runPerformanceTest();
    }
    
    const totalTime = Date.now() - startTime;
    
    log('═'.repeat(60), 'success');
    log(`🎉 Build optimization completed successfully in ${totalTime}ms!`, 'success');
    log('═'.repeat(60), 'success');
    
    // Show final stats
    const distStats = await analyzeDirectory(CONFIG.buildDir);
    log(`📦 Total bundle size: ${formatBytes(distStats.totalSize)}`, 'info');
    log(`📁 Build directory: ${CONFIG.buildDir}`, 'info');
    log(`📊 Reports directory: ${CONFIG.reportDir}`, 'info');
    
  } catch (error) {
    log(`Build optimization failed: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Run the optimization
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    log(`Unexpected error: ${error.message}`, 'error');
    process.exit(1);
  });
}