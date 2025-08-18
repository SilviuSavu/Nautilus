import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react({
    fastRefresh: false
  })],
  esbuild: {
    target: 'es2015',
    jsxFactory: 'React.createElement',
    jsxFragment: 'React.Fragment'
  },
  base: '/',
  server: {
    host: '0.0.0.0',
    port: 3000,
    open: false,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000',
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
    polyfillModulePreload: false,
    outDir: 'dist',
    sourcemap: true,
  },
  define: {
    'process.env': {},
  }
})