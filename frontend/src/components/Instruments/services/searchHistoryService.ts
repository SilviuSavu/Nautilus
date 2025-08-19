export interface SearchHistoryItem {
  query: string
  timestamp: string
  resultsCount: number
  filters?: {
    assetClasses?: string[]
    venues?: string[]
    currencies?: string[]
  }
}

class SearchHistoryService {
  private readonly HISTORY_KEY = 'instrument-search-history'
  private readonly MAX_HISTORY_ITEMS = 50

  // Add search query to history
  addToHistory(query: string, resultsCount: number, filters?: SearchHistoryItem['filters']): void {
    if (!query.trim()) return

    try {
      const history = this.getHistory()
      
      // Remove existing entry with same query and filters (to move it to top)
      const existingIndex = history.findIndex(item => 
        item.query === query.trim() && 
        JSON.stringify(item.filters || {}) === JSON.stringify(filters || {})
      )
      
      if (existingIndex >= 0) {
        history.splice(existingIndex, 1)
      }

      // Add new entry at the beginning
      const newItem: SearchHistoryItem = {
        query: query.trim(),
        timestamp: new Date().toISOString(),
        resultsCount,
        filters: filters && (
          (filters.assetClasses && filters.assetClasses.length > 0) ||
          (filters.venues && filters.venues.length > 0) ||
          (filters.currencies && filters.currencies.length > 0)
        ) ? filters : undefined
      }

      history.unshift(newItem)

      // Keep only the most recent items
      const trimmedHistory = history.slice(0, this.MAX_HISTORY_ITEMS)

      localStorage.setItem(this.HISTORY_KEY, JSON.stringify(trimmedHistory))
    } catch (error) {
      console.warn('Failed to add search to history:', error)
    }
  }

  // Get search history
  getHistory(): SearchHistoryItem[] {
    try {
      const history = localStorage.getItem(this.HISTORY_KEY)
      return history ? JSON.parse(history) : []
    } catch (error) {
      console.warn('Failed to load search history:', error)
      return []
    }
  }

  // Get recent searches (last N items)
  getRecentSearches(limit: number = 10): SearchHistoryItem[] {
    return this.getHistory().slice(0, limit)
  }

  // Get popular searches (most frequently used)
  getPopularSearches(limit: number = 10): SearchHistoryItem[] {
    const history = this.getHistory()
    
    // Group by query and count occurrences
    const queryCount = new Map<string, { item: SearchHistoryItem; count: number }>()
    
    history.forEach(item => {
      const existing = queryCount.get(item.query)
      if (existing) {
        existing.count++
        // Keep the most recent timestamp
        if (new Date(item.timestamp) > new Date(existing.item.timestamp)) {
          existing.item = item
        }
      } else {
        queryCount.set(item.query, { item, count: 1 })
      }
    })

    // Sort by count (descending) and return top items
    return Array.from(queryCount.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, limit)
      .map(entry => entry.item)
  }

  // Search history by query
  searchHistory(searchQuery: string): SearchHistoryItem[] {
    const history = this.getHistory()
    const normalizedQuery = searchQuery.toLowerCase()
    
    return history.filter(item => 
      item.query.toLowerCase().includes(normalizedQuery)
    )
  }

  // Remove item from history
  removeFromHistory(query: string, filters?: SearchHistoryItem['filters']): void {
    try {
      const history = this.getHistory()
      const filteredHistory = history.filter(item => 
        !(item.query === query && 
          JSON.stringify(item.filters || {}) === JSON.stringify(filters || {}))
      )
      
      localStorage.setItem(this.HISTORY_KEY, JSON.stringify(filteredHistory))
    } catch (error) {
      console.warn('Failed to remove item from search history:', error)
    }
  }

  // Clear all search history
  clearHistory(): void {
    try {
      localStorage.removeItem(this.HISTORY_KEY)
    } catch (error) {
      console.warn('Failed to clear search history:', error)
    }
  }

  // Get history statistics
  getHistoryStats(): {
    totalSearches: number
    uniqueQueries: number
    mostPopularQuery?: string
    lastSearchDate?: string
  } {
    const history = this.getHistory()
    const uniqueQueries = new Set(history.map(item => item.query))
    
    let mostPopularQuery: string | undefined
    let maxCount = 0
    const queryCounts = new Map<string, number>()
    
    history.forEach(item => {
      const count = (queryCounts.get(item.query) || 0) + 1
      queryCounts.set(item.query, count)
      
      if (count > maxCount) {
        maxCount = count
        mostPopularQuery = item.query
      }
    })

    return {
      totalSearches: history.length,
      uniqueQueries: uniqueQueries.size,
      mostPopularQuery,
      lastSearchDate: history.length > 0 ? history[0].timestamp : undefined
    }
  }

  // Export search history (for debugging or user export)
  exportHistory(): string {
    return JSON.stringify(this.getHistory(), null, 2)
  }

  // Import search history
  importHistory(historyJson: string): boolean {
    try {
      const imported = JSON.parse(historyJson) as SearchHistoryItem[]
      
      // Validate structure
      if (!Array.isArray(imported)) return false
      
      // Merge with existing history
      const existing = this.getHistory()
      const combined = [...imported, ...existing]
      
      // Remove duplicates and limit size
      const uniqueItems = new Map<string, SearchHistoryItem>()
      combined.forEach(item => {
        const key = `${item.query}-${JSON.stringify(item.filters || {})}`
        if (!uniqueItems.has(key) || 
            new Date(item.timestamp) > new Date(uniqueItems.get(key)!.timestamp)) {
          uniqueItems.set(key, item)
        }
      })
      
      const finalHistory = Array.from(uniqueItems.values())
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, this.MAX_HISTORY_ITEMS)
      
      localStorage.setItem(this.HISTORY_KEY, JSON.stringify(finalHistory))
      return true
    } catch (error) {
      console.warn('Failed to import search history:', error)
      return false
    }
  }
}

export const searchHistoryService = new SearchHistoryService()