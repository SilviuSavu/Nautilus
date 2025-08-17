import React, { useState } from 'react'
import { useChartStore } from './hooks/useChartStore'
import { IndicatorConfig } from './types/chartTypes'

interface IndicatorPanelProps {
  className?: string
}

const INDICATOR_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
  '#FD79A8', '#B8860B', '#8E44AD', '#E17055', '#00B894'
]

export const IndicatorPanel: React.FC<IndicatorPanelProps> = ({ className = '' }) => {
  const { 
    indicators, 
    addIndicator, 
    removeIndicator, 
    updateIndicator 
  } = useChartStore()

  const [isExpanded, setIsExpanded] = useState(false)
  const [newIndicator, setNewIndicator] = useState({
    type: 'SMA' as const,
    period: 20
  })

  const handleAddIndicator = () => {
    const indicator: IndicatorConfig = {
      id: `${newIndicator.type}_${newIndicator.period}_${Date.now()}`,
      type: newIndicator.type,
      period: newIndicator.period,
      color: INDICATOR_COLORS[indicators.length % INDICATOR_COLORS.length],
      visible: true
    }
    
    addIndicator(indicator)
    
    // Reset form
    setNewIndicator({
      type: 'SMA',
      period: 20
    })
  }

  const handleToggleVisibility = (indicatorId: string) => {
    const indicator = indicators.find(ind => ind.id === indicatorId)
    if (indicator) {
      updateIndicator(indicatorId, { visible: !indicator.visible })
    }
  }

  const handleColorChange = (indicatorId: string, color: string) => {
    updateIndicator(indicatorId, { color })
  }

  const handlePeriodChange = (indicatorId: string, period: number) => {
    updateIndicator(indicatorId, { period })
  }

  return (
    <div className={`indicator-panel ${className}`}>
      <div className="indicator-panel-header">
        <button
          className="indicator-panel-toggle"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-label={isExpanded ? 'Collapse indicators' : 'Expand indicators'}
        >
          <span className="indicator-panel-title">
            📊 Indicators ({indicators.length})
          </span>
          <span className={`indicator-panel-arrow ${isExpanded ? 'expanded' : ''}`}>
            ▼
          </span>
        </button>
      </div>

      {isExpanded && (
        <div className="indicator-panel-content">
          {/* Add New Indicator Form */}
          <div className="indicator-add-form">
            <h4>Add Indicator</h4>
            <div className="indicator-form-row">
              <select
                value={newIndicator.type}
                onChange={(e) => setNewIndicator(prev => ({ 
                  ...prev, 
                  type: e.target.value as 'SMA' | 'EMA' 
                }))}
                className="indicator-type-select"
              >
                <option value="SMA">Simple Moving Average (SMA)</option>
                <option value="EMA">Exponential Moving Average (EMA)</option>
              </select>
            </div>
            <div className="indicator-form-row">
              <label htmlFor="period-input">Period:</label>
              <input
                id="period-input"
                type="number"
                min="1"
                max="200"
                value={newIndicator.period}
                onChange={(e) => setNewIndicator(prev => ({ 
                  ...prev, 
                  period: parseInt(e.target.value) || 1 
                }))}
                className="indicator-period-input"
              />
            </div>
            <button
              onClick={handleAddIndicator}
              className="indicator-add-button"
            >
              Add {newIndicator.type}({newIndicator.period})
            </button>
          </div>

          {/* Active Indicators List */}
          {indicators.length > 0 && (
            <div className="indicator-list">
              <h4>Active Indicators</h4>
              {indicators.map(indicator => (
                <div key={indicator.id} className="indicator-item">
                  <div className="indicator-info">
                    <span className="indicator-name">
                      {indicator.type}({indicator.period})
                    </span>
                    <div className="indicator-controls">
                      <input
                        type="color"
                        value={indicator.color}
                        onChange={(e) => handleColorChange(indicator.id, e.target.value)}
                        className="indicator-color-picker"
                        title="Change color"
                      />
                      <input
                        type="number"
                        min="1"
                        max="200"
                        value={indicator.period}
                        onChange={(e) => handlePeriodChange(
                          indicator.id, 
                          parseInt(e.target.value) || 1
                        )}
                        className="indicator-period-edit"
                        title="Edit period"
                      />
                      <button
                        onClick={() => handleToggleVisibility(indicator.id)}
                        className={`indicator-visibility-toggle ${indicator.visible ? 'visible' : 'hidden'}`}
                        title={indicator.visible ? 'Hide indicator' : 'Show indicator'}
                      >
                        {indicator.visible ? '👁️' : '👁️‍🗨️'}
                      </button>
                      <button
                        onClick={() => removeIndicator(indicator.id)}
                        className="indicator-remove-button"
                        title="Remove indicator"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {indicators.length === 0 && (
            <div className="indicator-empty-state">
              <p>No indicators added yet. Add your first technical indicator above.</p>
            </div>
          )}
        </div>
      )}

      <style>{`
        .indicator-panel {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          margin: 8px 0;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .indicator-panel-header {
          padding: 0;
        }

        .indicator-panel-toggle {
          width: 100%;
          background: transparent;
          border: none;
          color: #e0e0e0;
          padding: 12px 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.2s;
        }

        .indicator-panel-toggle:hover {
          background: #2a2a2a;
        }

        .indicator-panel-title {
          font-weight: 500;
        }

        .indicator-panel-arrow {
          transition: transform 0.2s;
          font-size: 12px;
        }

        .indicator-panel-arrow.expanded {
          transform: rotate(180deg);
        }

        .indicator-panel-content {
          border-top: 1px solid #333;
          padding: 16px;
        }

        .indicator-add-form {
          margin-bottom: 20px;
        }

        .indicator-add-form h4 {
          margin: 0 0 12px 0;
          color: #e0e0e0;
          font-size: 13px;
          font-weight: 500;
        }

        .indicator-form-row {
          margin-bottom: 8px;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .indicator-form-row label {
          color: #b0b0b0;
          font-size: 12px;
          min-width: 50px;
        }

        .indicator-type-select,
        .indicator-period-input {
          background: #2a2a2a;
          border: 1px solid #444;
          color: #e0e0e0;
          padding: 6px 8px;
          border-radius: 3px;
          font-size: 12px;
        }

        .indicator-type-select {
          flex: 1;
        }

        .indicator-period-input {
          width: 60px;
        }

        .indicator-add-button {
          background: #4ECDC4;
          color: #000;
          border: none;
          padding: 8px 12px;
          border-radius: 3px;
          cursor: pointer;
          font-size: 12px;
          font-weight: 500;
          margin-top: 8px;
          transition: background-color 0.2s;
        }

        .indicator-add-button:hover {
          background: #45B7D1;
        }

        .indicator-list h4 {
          margin: 0 0 12px 0;
          color: #e0e0e0;
          font-size: 13px;
          font-weight: 500;
        }

        .indicator-item {
          margin-bottom: 8px;
        }

        .indicator-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px;
          background: #2a2a2a;
          border-radius: 3px;
        }

        .indicator-name {
          color: #e0e0e0;
          font-size: 12px;
          font-weight: 500;
        }

        .indicator-controls {
          display: flex;
          gap: 6px;
          align-items: center;
        }

        .indicator-color-picker {
          width: 20px;
          height: 20px;
          border: none;
          border-radius: 2px;
          cursor: pointer;
        }

        .indicator-period-edit {
          width: 40px;
          background: #1a1a1a;
          border: 1px solid #444;
          color: #e0e0e0;
          padding: 2px 4px;
          border-radius: 2px;
          font-size: 11px;
          text-align: center;
        }

        .indicator-visibility-toggle {
          background: transparent;
          border: none;
          cursor: pointer;
          font-size: 14px;
          padding: 2px;
        }

        .indicator-visibility-toggle.hidden {
          opacity: 0.5;
        }

        .indicator-remove-button {
          background: #ff4757;
          color: white;
          border: none;
          width: 18px;
          height: 18px;
          border-radius: 2px;
          cursor: pointer;
          font-size: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .indicator-remove-button:hover {
          background: #ff3742;
        }

        .indicator-empty-state {
          text-align: center;
          color: #888;
          font-size: 12px;
          margin: 20px 0;
        }
      `}</style>
    </div>
  )
}