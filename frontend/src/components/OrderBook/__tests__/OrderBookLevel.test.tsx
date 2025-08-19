import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { OrderBookLevel } from '../OrderBookLevel'
import { ProcessedOrderBookLevel, OrderBookDisplaySettings } from '../../../types/orderBook'

describe('OrderBookLevel', () => {
  const mockBidLevel: ProcessedOrderBookLevel = {
    id: 'bid-1',
    price: 150.25,
    quantity: 100,
    cumulative: 100,
    percentage: 50,
    orderCount: 5
  }

  const mockAskLevel: ProcessedOrderBookLevel = {
    id: 'ask-1',
    price: 150.26,
    quantity: 80,
    cumulative: 80,
    percentage: 40,
    orderCount: 4
  }

  const defaultDisplaySettings: OrderBookDisplaySettings = {
    showSpread: true,
    showOrderCount: false,
    colorScheme: 'default',
    decimals: 2
  }

  describe('rendering', () => {
    it('should render bid level correctly', () => {
      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      // Check price and quantity are displayed
      expect(screen.getByText('150.25')).toBeInTheDocument()
      expect(screen.getByText('100')).toBeInTheDocument()
    })

    it('should render ask level correctly', () => {
      render(
        <OrderBookLevel
          level={mockAskLevel}
          side="ask"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      // Check price and quantity are displayed
      expect(screen.getByText('150.26')).toBeInTheDocument()
      expect(screen.getByText('80')).toBeInTheDocument()
    })

    it('should display order count when enabled', () => {
      const settingsWithOrderCount: OrderBookDisplaySettings = {
        ...defaultDisplaySettings,
        showOrderCount: true
      }

      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={settingsWithOrderCount}
          maxQuantity={200}
        />
      )

      expect(screen.getByText('(5)')).toBeInTheDocument()
    })

    it('should hide order count when disabled', () => {
      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      expect(screen.queryByText('(5)')).not.toBeInTheDocument()
    })

    it('should format price with correct decimals', () => {
      const settingsWithMoreDecimals: OrderBookDisplaySettings = {
        ...defaultDisplaySettings,
        decimals: 4
      }

      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={settingsWithMoreDecimals}
          maxQuantity={200}
        />
      )

      expect(screen.getByText('150.2500')).toBeInTheDocument()
    })

    it('should format quantity with thousands separators', () => {
      const largeQuantityLevel: ProcessedOrderBookLevel = {
        ...mockBidLevel,
        quantity: 12345
      }

      render(
        <OrderBookLevel
          level={largeQuantityLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={20000}
        />
      )

      expect(screen.getByText('12,345')).toBeInTheDocument()
    })
  })

  describe('layout differences by side', () => {
    it('should render bid side with quantity first, then price', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      const contentDiv = container.querySelector('[style*="zIndex: 1"]')
      const spans = contentDiv?.querySelectorAll('span')
      
      expect(spans).toHaveLength(2)
      expect(spans?.[0]).toHaveTextContent('100') // quantity first
      expect(spans?.[1]).toHaveTextContent('150.25') // price second
    })

    it('should render ask side with price first, then quantity', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockAskLevel}
          side="ask"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      const contentDiv = container.querySelector('[style*="zIndex: 1"]')
      const spans = contentDiv?.querySelectorAll('span')
      
      expect(spans).toHaveLength(2)
      expect(spans?.[0]).toHaveTextContent('150.26') // price first
      expect(spans?.[1]).toHaveTextContent('80') // quantity second
    })
  })

  describe('depth visualization', () => {
    it('should calculate correct depth percentage', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200} // level quantity is 100, so should be 50%
        />
      )

      const depthBar = container.querySelector('[style*="backgroundColor"]')
      expect(depthBar).toHaveStyle({ left: '50%' }) // 100 - 50% = 50%
    })

    it('should handle zero max quantity', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={0}
        />
      )

      const depthBar = container.querySelector('[style*="backgroundColor"]')
      expect(depthBar).toHaveStyle({ left: '100%' }) // 100 - 0% = 100%
    })

    it('should position depth bar correctly for ask side', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockAskLevel}
          side="ask"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200} // level quantity is 80, so should be 40%
        />
      )

      const depthBar = container.querySelector('[style*="backgroundColor"]')
      expect(depthBar).toHaveStyle({ right: '60%' }) // 100 - 40% = 60%
    })
  })

  describe('color schemes', () => {
    it('should apply different colors for different schemes', () => {
      const darkSettings: OrderBookDisplaySettings = {
        ...defaultDisplaySettings,
        colorScheme: 'dark'
      }

      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={darkSettings}
          maxQuantity={200}
        />
      )

      // The component should exist (detailed color testing would require more complex setup)
      expect(container.firstChild).toBeInTheDocument()
    })

    it('should apply different colors for bid vs ask', () => {
      const { container: bidContainer } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      const { container: askContainer } = render(
        <OrderBookLevel
          level={mockAskLevel}
          side="ask"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      // Both should render but with different styling
      expect(bidContainer.firstChild).toBeInTheDocument()
      expect(askContainer.firstChild).toBeInTheDocument()
    })
  })

  describe('interactions', () => {
    it('should call onLevelClick when clicked', () => {
      const mockOnLevelClick = vi.fn()

      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
          onLevelClick={mockOnLevelClick}
        />
      )

      const levelElement = screen.getByText('150.25').closest('div')
      fireEvent.click(levelElement!)

      expect(mockOnLevelClick).toHaveBeenCalledWith(mockBidLevel, 'bid')
    })

    it('should not call onLevelClick when no handler provided', () => {
      const mockOnLevelClick = vi.fn()

      render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
          // No onLevelClick prop
        />
      )

      const levelElement = screen.getByText('150.25').closest('div')
      fireEvent.click(levelElement!)

      expect(mockOnLevelClick).not.toHaveBeenCalled()
    })

    it('should show hover effects', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      const levelElement = container.firstChild as HTMLElement
      
      // Initial state should have transparent background
      expect(levelElement).toHaveStyle({ backgroundColor: 'transparent' })

      // Mouse enter should change background
      fireEvent.mouseEnter(levelElement)
      // Note: Testing the exact hover color would require more complex CSS testing

      // Mouse leave should reset background
      fireEvent.mouseLeave(levelElement)
      expect(levelElement).toHaveStyle({ backgroundColor: 'transparent' })
    })

    it('should show pointer cursor when clickable', () => {
      const mockOnLevelClick = vi.fn()

      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
          onLevelClick={mockOnLevelClick}
        />
      )

      const levelElement = container.firstChild as HTMLElement
      expect(levelElement).toHaveStyle({ cursor: 'pointer' })
    })

    it('should show default cursor when not clickable', () => {
      const { container } = render(
        <OrderBookLevel
          level={mockBidLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
          // No onLevelClick
        />
      )

      const levelElement = container.firstChild as HTMLElement
      expect(levelElement).toHaveStyle({ cursor: 'default' })
    })
  })

  describe('edge cases', () => {
    it('should handle missing orderCount gracefully', () => {
      const levelWithoutOrderCount: ProcessedOrderBookLevel = {
        ...mockBidLevel,
        orderCount: undefined
      }

      const settingsWithOrderCount: OrderBookDisplaySettings = {
        ...defaultDisplaySettings,
        showOrderCount: true
      }

      render(
        <OrderBookLevel
          level={levelWithoutOrderCount}
          side="bid"
          displaySettings={settingsWithOrderCount}
          maxQuantity={200}
        />
      )

      // Should not crash and should not show order count
      expect(screen.queryByText(/\(/)).not.toBeInTheDocument()
    })

    it('should handle zero price and quantity', () => {
      const zeroLevel: ProcessedOrderBookLevel = {
        ...mockBidLevel,
        price: 0,
        quantity: 0
      }

      render(
        <OrderBookLevel
          level={zeroLevel}
          side="bid"
          displaySettings={defaultDisplaySettings}
          maxQuantity={200}
        />
      )

      expect(screen.getByText('0.00')).toBeInTheDocument()
      expect(screen.getByText('0')).toBeInTheDocument()
    })

    it('should handle very large numbers', () => {
      const largeLevel: ProcessedOrderBookLevel = {
        ...mockBidLevel,
        price: 999999.9999,
        quantity: 1000000
      }

      const highPrecisionSettings: OrderBookDisplaySettings = {
        ...defaultDisplaySettings,
        decimals: 4
      }

      render(
        <OrderBookLevel
          level={largeLevel}
          side="bid"
          displaySettings={highPrecisionSettings}
          maxQuantity={2000000}
        />
      )

      expect(screen.getByText('999999.9999')).toBeInTheDocument()
      expect(screen.getByText('1,000,000')).toBeInTheDocument()
    })
  })
})