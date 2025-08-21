import React, { useState } from 'react'
import { useChartStore } from './hooks/useChartStore'

interface ChartControlsProps {
  onZoomIn?: () => void
  onZoomOut?: () => void
  onZoomFit?: () => void
  onToggleFullscreen?: () => void
  onResetChart?: () => void
  className?: string
}

export const ChartControls: React.FC<ChartControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onZoomFit,
  onToggleFullscreen,
  onResetChart,
  className = ''
}) => {
  const { 
    settings, 
    updateSettings, 
    realTimeUpdates, 
    toggleRealTimeUpdates 
  } = useChartStore()

  const [showSettings, setShowSettings] = useState(false)

  const handleToggleVolume = () => {
    updateSettings({ showVolume: !settings.showVolume })
  }

  const handleToggleCrosshair = () => {
    updateSettings({ crosshair: !settings.crosshair })
  }

  const handleToggleGrid = () => {
    updateSettings({ grid: !settings.grid })
  }

  const handleTimezoneChange = (timezone: string) => {
    updateSettings({ timezone })
  }

  // Keyboard shortcuts handler
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case '+':
          case '=':
            event.preventDefault()
            onZoomIn?.()
            break
          case '-':
            event.preventDefault()
            onZoomOut?.()
            break
          case '0':
            event.preventDefault()
            onZoomFit?.()
            break
        }
      } else {
        switch (event.key.toLowerCase()) {
          case 'f':
            if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
              event.preventDefault()
              onToggleFullscreen?.()
            }
            break
          case 'r':
            if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
              event.preventDefault()
              onResetChart?.()
            }
            break
          case 'v':
            if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
              event.preventDefault()
              handleToggleVolume()
            }
            break
          case 'c':
            if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
              event.preventDefault()
              handleToggleCrosshair()
            }
            break
          case 'g':
            if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
              event.preventDefault()
              handleToggleGrid()
            }
            break
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onZoomIn, onZoomOut, onZoomFit, onToggleFullscreen, onResetChart])

  return (
    <div className={`chart-controls ${className}`}>
      {/* Main Control Buttons */}
      <div className="control-group">
        <button
          onClick={onZoomIn}
          className="control-button"
          title="Zoom In (Ctrl/Cmd + +)"
        >
          🔍+
        </button>
        <button
          onClick={onZoomOut}
          className="control-button"
          title="Zoom Out (Ctrl/Cmd + -)"
        >
          🔍-
        </button>
        <button
          onClick={onZoomFit}
          className="control-button"
          title="Zoom to Fit (Ctrl/Cmd + 0)"
        >
          📏
        </button>
        <button
          onClick={onResetChart}
          className="control-button"
          title="Reset Chart (R)"
        >
          🔄
        </button>
      </div>

      {/* View Controls */}
      <div className="control-group">
        <button
          onClick={handleToggleVolume}
          className={`control-button ${settings.showVolume ? 'active' : ''}`}
          title="Toggle Volume (V)"
        >
          📊
        </button>
        <button
          onClick={handleToggleCrosshair}
          className={`control-button ${settings.crosshair ? 'active' : ''}`}
          title="Toggle Crosshair (C)"
        >
          ✛
        </button>
        <button
          onClick={handleToggleGrid}
          className={`control-button ${settings.grid ? 'active' : ''}`}
          title="Toggle Grid (G)"
        >
          ⊞
        </button>
      </div>

      {/* Real-time Updates */}
      <div className="control-group">
        <button
          onClick={toggleRealTimeUpdates}
          className={`control-button ${realTimeUpdates ? 'active real-time' : ''}`}
          title="Toggle Real-time Updates"
        >
          {realTimeUpdates ? '🟢' : '⏸️'} Live
        </button>
      </div>

      {/* Settings Toggle */}
      <div className="control-group">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`control-button ${showSettings ? 'active' : ''}`}
          title="Chart Settings"
        >
          ⚙️
        </button>
        <button
          onClick={onToggleFullscreen}
          className="control-button"
          title="Toggle Fullscreen (F)"
        >
          ⛶
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel">
          <div className="settings-section">
            <h4>Chart Settings</h4>
            
            <div className="setting-row">
              <label>Timezone:</label>
              <select
                value={settings.timezone}
                onChange={(e) => handleTimezoneChange(e.target.value)}
                className="timezone-select"
              >
                <option value="UTC">UTC</option>
                <option value="America/New_York">New York</option>
                <option value="America/Chicago">Chicago</option>
                <option value="America/Los_Angeles">Los Angeles</option>
                <option value="Europe/London">London</option>
                <option value="Europe/Frankfurt">Frankfurt</option>
                <option value="Asia/Tokyo">Tokyo</option>
                <option value="Asia/Hong_Kong">Hong Kong</option>
                <option value="Asia/Singapore">Singapore</option>
              </select>
            </div>

            <div className="setting-row">
              <label>
                <input
                  type="checkbox"
                  checked={settings.showVolume}
                  onChange={handleToggleVolume}
                />
                Show Volume
              </label>
            </div>

            <div className="setting-row">
              <label>
                <input
                  type="checkbox"
                  checked={settings.crosshair}
                  onChange={handleToggleCrosshair}
                />
                Show Crosshair
              </label>
            </div>

            <div className="setting-row">
              <label>
                <input
                  type="checkbox"
                  checked={settings.grid}
                  onChange={handleToggleGrid}
                />
                Show Grid
              </label>
            </div>
          </div>

          <div className="settings-section">
            <h4>Keyboard Shortcuts</h4>
            <div className="shortcuts-list">
              <div className="shortcut-item">
                <span className="shortcut-key">Ctrl/Cmd + +</span>
                <span className="shortcut-desc">Zoom In</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">Ctrl/Cmd + -</span>
                <span className="shortcut-desc">Zoom Out</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">Ctrl/Cmd + 0</span>
                <span className="shortcut-desc">Zoom to Fit</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">F</span>
                <span className="shortcut-desc">Fullscreen</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">R</span>
                <span className="shortcut-desc">Reset Chart</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">V</span>
                <span className="shortcut-desc">Toggle Volume</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">C</span>
                <span className="shortcut-desc">Toggle Crosshair</span>
              </div>
              <div className="shortcut-item">
                <span className="shortcut-key">G</span>
                <span className="shortcut-desc">Toggle Grid</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .chart-controls {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 8px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          position: relative;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .control-group {
          display: flex;
          gap: 4px;
          padding: 0 4px;
          border-right: 1px solid #333;
        }

        .control-group:last-child {
          border-right: none;
        }

        .control-button {
          background: #2a2a2a;
          border: 1px solid #444;
          color: #e0e0e0;
          padding: 6px 8px;
          border-radius: 3px;
          cursor: pointer;
          font-size: 12px;
          min-width: 32px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
          user-select: none;
        }

        .control-button:hover {
          background: #3a3a3a;
          border-color: #555;
        }

        .control-button.active {
          background: #4ECDC4;
          color: #000;
          border-color: #4ECDC4;
        }

        .control-button.real-time {
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.7; }
          100% { opacity: 1; }
        }

        .settings-panel {
          position: absolute;
          top: 100%;
          right: 0;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          padding: 16px;
          min-width: 300px;
          z-index: 1000;
          margin-top: 4px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .settings-section {
          margin-bottom: 16px;
        }

        .settings-section:last-child {
          margin-bottom: 0;
        }

        .settings-section h4 {
          margin: 0 0 12px 0;
          color: #e0e0e0;
          font-size: 13px;
          font-weight: 500;
          border-bottom: 1px solid #333;
          padding-bottom: 6px;
        }

        .setting-row {
          margin-bottom: 8px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }

        .setting-row label {
          color: #b0b0b0;
          font-size: 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
        }

        .timezone-select {
          background: #2a2a2a;
          border: 1px solid #444;
          color: #e0e0e0;
          padding: 4px 6px;
          border-radius: 3px;
          font-size: 12px;
        }

        .shortcuts-list {
          font-size: 11px;
        }

        .shortcut-item {
          display: flex;
          justify-content: space-between;
          margin-bottom: 4px;
          color: #888;
        }

        .shortcut-key {
          font-family: monospace;
          background: #2a2a2a;
          padding: 2px 4px;
          border-radius: 2px;
          color: #e0e0e0;
        }

        .shortcut-desc {
          color: #b0b0b0;
        }

        input[type="checkbox"] {
          margin: 0;
          width: 14px;
          height: 14px;
        }
      `}</style>
    </div>
  )
}