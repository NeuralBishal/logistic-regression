#!/usr/bin/env node
console.log('Running Vercel build...');
console.log('Building API server...');

const { execSync } = require('child_process');
try {
  // Build the API server
  execSync('cd artifacts/api-server && node build.mjs', { 
    stdio: 'inherit',
    env: { ...process.env, SKIP_TSC: 'true' }
  });
  console.log('Build completed successfully!');
} catch (error) {
  console.error('Build failed:', error);
  process.exit(1);
}
