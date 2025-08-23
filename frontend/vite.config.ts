import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react({
    fastRefresh: false
  })],
  esbuild: {
    target: 'es2015',
    jsxFactory: 'React.createElement',
    jsxFragment: 'React.Fragment',
    logOverride: { 'this-is-undefined-in-esm': 'silent' },
    ignoreAnnotations: true
  },
  base: '/',
  server: {
    host: '0.0.0.0',
    port: 3000,
    open: false,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
        changeOrigin: true,
      },
      '/ws/messagebus': {
        target: 'ws://localhost:8001',
        ws: true,
        changeOrigin: true,
      }
    },
    watch: {
      usePolling: true,
    }
  },
  build: {
    target: 'es2015',
    modulePreload: {
      polyfill: false
    },
    outDir: 'dist',
    sourcemap: true,
  },
  define: {
    'process.env': {
      NODE_ENV: JSON.stringify(process.env.NODE_ENV || 'development'),
    },
  },
  envPrefix: 'VITE_'
})