import { useEffect, useCallback } from 'react'
import { Instrument } from '../types/instrumentTypes'

interface KeyboardShortcutsConfig {
  searchResults: Instrument[]
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  onInstrumentSelect: (instrument: Instrument) => void
  onToggleFilters?: () => void
  onToggleSettings?: () => void
  onClearSearch?: () => void
  onFocusSearch?: () => void
  searchInputRef?: React.RefObject<HTMLInputElement>
  isSearchFocused: boolean
  maxResults: number
}

export const useKeyboardShortcuts = ({
  searchResults,
  selectedIndex,
  setSelectedIndex,
  onInstrumentSelect,
  onToggleFilters,
  onToggleSettings,
  onClearSearch,
  onFocusSearch,
  searchInputRef,
  isSearchFocused,
  maxResults
}: KeyboardShortcutsConfig) => {
  
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Global shortcuts (work when search is not focused)
    if (!isSearchFocused) {
      // Cmd/Ctrl + K to focus search
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault()
        onFocusSearch?.()
        return
      }
      
      // / to focus search (like GitHub)
      if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        onFocusSearch?.()
        return
      }
    }
    
    // Search-focused shortcuts
    if (isSearchFocused && searchResults.length > 0) {
      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault()
          setSelectedIndex(Math.min(selectedIndex + 1, searchResults.length - 1))
          break
          
        case 'ArrowUp':
          event.preventDefault()
          setSelectedIndex(Math.max(selectedIndex - 1, -1))
          break
          
        case 'Enter':
          event.preventDefault()
          if (selectedIndex >= 0 && selectedIndex < searchResults.length) {
            onInstrumentSelect(searchResults[selectedIndex])
          }
          break
          
        case 'Escape':
          event.preventDefault()
          onClearSearch?.()
          searchInputRef?.current?.blur()
          break
          
        case 'Tab':
          if (event.shiftKey) {
            // Shift+Tab - previous result (alternative to ArrowUp)
            event.preventDefault()
            setSelectedIndex(Math.max(selectedIndex - 1, -1))
          } else {
            // Tab - next result (alternative to ArrowDown)
            event.preventDefault()
            setSelectedIndex(Math.min(selectedIndex + 1, searchResults.length - 1))
          }
          break
          
        default:
          break
      }
    }
    
    // Global shortcuts that work regardless of focus
    // Cmd/Ctrl + Shift shortcuts
    if ((event.metaKey || event.ctrlKey) && event.shiftKey) {
      switch (event.key) {
        case 'F': // Cmd/Ctrl + Shift + F for filters
          event.preventDefault()
          onToggleFilters?.()
          break
          
        case 'S': // Cmd/Ctrl + Shift + S for settings
          event.preventDefault()
          onToggleSettings?.()
          break
          
        default:
          break
      }
    }
    
    // Number shortcuts for quick selection (1-9)
    if (isSearchFocused && event.key >= '1' && event.key <= '9' && !event.metaKey && !event.ctrlKey) {
      const index = parseInt(event.key) - 1
      if (index < searchResults.length) {
        event.preventDefault()
        onInstrumentSelect(searchResults[index])
      }
    }
    
  }, [
    searchResults,
    selectedIndex,
    setSelectedIndex,
    onInstrumentSelect,
    onToggleFilters,
    onToggleSettings,
    onClearSearch,
    onFocusSearch,
    searchInputRef,
    isSearchFocused
  ])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [handleKeyDown])

  // Return helper functions and state
  return {
    selectedIndex,
    setSelectedIndex,
    // Helper to get selected instrument
    getSelectedInstrument: () => {
      if (selectedIndex >= 0 && selectedIndex < searchResults.length) {
        return searchResults[selectedIndex]
      }
      return null
    }
  }
}

export default useKeyboardShortcuts