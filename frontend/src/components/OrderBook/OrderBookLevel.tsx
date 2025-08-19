import React, { memo } from 'react'
import { ProcessedOrderBookLevel, OrderBookDisplaySettings } from '../../types/orderBook'

interface OrderBookLevelProps {
  level: ProcessedOrderBookLevel
  side: 'bid' | 'ask'
  displaySettings: OrderBookDisplaySettings
  maxQuantity: number
  onLevelClick?: (level: ProcessedOrderBookLevel, side: 'bid' | 'ask') => void
  className?: string
}

export const OrderBookLevel: React.FC<OrderBookLevelProps> = memo(({
  level,
  side,
  displaySettings,
  maxQuantity,
  onLevelClick,
  className
}) => {
  const { decimals, showOrderCount, colorScheme } = displaySettings

  // Calculate depth bar width percentage
  const depthPercentage = maxQuantity > 0 ? (level.quantity / maxQuantity) * 100 : 0

  // Color scheme based on side and theme
  const getColors = () => {
    const themes = {
      default: {
        bid: {
          background: '#e6f7e6',
          border: '#4caf50',
          text: '#2e7d32',
          depth: '#4caf5020'
        },
        ask: {
          background: '#ffe6e6',
          border: '#f44336',
          text: '#c62828',
          depth: '#f4433620'
        }
      },
      dark: {
        bid: {
          background: '#1a2e1a',
          border: '#4caf50',
          text: '#81c784',
          depth: '#4caf5030'
        },
        ask: {
          background: '#2e1a1a',
          border: '#f44336',
          text: '#ef5350',
          depth: '#f4433630'
        }
      },
      light: {
        bid: {
          background: '#f1f8e9',
          border: '#8bc34a',
          text: '#33691e',
          depth: '#8bc34a15'
        },
        ask: {
          background: '#fce4ec',
          border: '#e91e63',
          text: '#ad1457',
          depth: '#e91e6315'
        }
      }
    }
    return themes[colorScheme][side]
  }

  const colors = getColors()

  const handleClick = () => {
    if (onLevelClick) {
      onLevelClick(level, side)
    }
  }

  const containerStyle: React.CSSProperties = {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '2px 8px',
    fontSize: '12px',
    fontFamily: 'monospace',
    cursor: onLevelClick ? 'pointer' : 'default',
    minHeight: '20px',
    borderLeft: side === 'bid' ? `2px solid ${colors.border}` : 'none',
    borderRight: side === 'ask' ? `2px solid ${colors.border}` : 'none',
    color: colors.text,
    backgroundColor: 'transparent',
    transition: 'background-color 0.2s ease'
  }

  const depthBarStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: side === 'bid' ? `${100 - depthPercentage}%` : 0,
    right: side === 'ask' ? `${100 - depthPercentage}%` : 0,
    backgroundColor: colors.depth,
    transition: 'all 0.3s ease',
    zIndex: 0
  }

  const contentStyle: React.CSSProperties = {
    position: 'relative',
    zIndex: 1,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%'
  }

  return (
    <div
      className={className}
      style={containerStyle}
      onClick={handleClick}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = colors.background
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent'
      }}
    >
      {/* Depth visualization bar */}
      <div style={depthBarStyle} />
      
      {/* Content */}
      <div style={contentStyle}>
        {side === 'bid' ? (
          <>
            {/* Bid side: quantity, price */}
            <span style={{ fontWeight: 'bold', minWidth: '60px', textAlign: 'right' }}>
              {level.quantity.toLocaleString(undefined, {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
              })}
            </span>
            <span style={{ fontWeight: 'bold', minWidth: '80px', textAlign: 'right' }}>
              {level.price.toFixed(decimals)}
            </span>
          </>
        ) : (
          <>
            {/* Ask side: price, quantity */}
            <span style={{ fontWeight: 'bold', minWidth: '80px', textAlign: 'left' }}>
              {level.price.toFixed(decimals)}
            </span>
            <span style={{ fontWeight: 'bold', minWidth: '60px', textAlign: 'left' }}>
              {level.quantity.toLocaleString(undefined, {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
              })}
            </span>
          </>
        )}
        
        {/* Order count (optional) */}
        {showOrderCount && level.orderCount && (
          <span style={{ 
            fontSize: '10px', 
            opacity: 0.7,
            minWidth: '30px',
            textAlign: 'center'
          }}>
            ({level.orderCount})
          </span>
        )}
      </div>
    </div>
  )
})

OrderBookLevel.displayName = 'OrderBookLevel'