import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock environment variables for tests
Object.defineProperty(window, 'import', {
  value: {
    meta: {
      env: {
        MODE: 'test',
        VITE_API_BASE_URL: 'http://localhost:8000',
        VITE_WS_URL: 'ws://localhost:8000/ws',
        VITE_DEBUG: 'true'
      }
    }
  }
})

// Mock fetch for tests
;(globalThis as any).fetch = vi.fn()

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock window.getComputedStyle
Object.defineProperty(window, 'getComputedStyle', {
  value: () => ({
    getPropertyValue: () => '',
  }),
})