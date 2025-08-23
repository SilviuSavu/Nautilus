"""
Data.gov Instrument Provider
============================

Maps Data.gov datasets to NautilusTrader instruments for trading integration.
Follows EDGAR instrument provider patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from functools import lru_cache

from .api_client import DatagovAPIClient
from .config import DatagovInstrumentConfig
from .data_types import DatagovDataset, DatasetCategory, DatasetFrequency
from .utils import assess_trading_relevance, sanitize_dataset_name

logger = logging.getLogger(__name__)


class DatagovInstrumentProvider:
    """
    Provides instrument definitions from Data.gov datasets.
    
    Maps datasets to tradeable instruments for NautilusTrader integration:
    - Dataset metadata -> Instrument properties
    - Real-time update monitoring
    - Trading relevance scoring
    - Category-based organization
    """
    
    def __init__(self, api_client: DatagovAPIClient, config: DatagovInstrumentConfig):
        self.api_client = api_client
        self.config = config
        self._datasets: Dict[str, DatagovDataset] = {}
        self._dataset_index: Dict[str, str] = {}  # title -> id mapping
        self._category_index: Dict[DatasetCategory, List[str]] = {}
        self._is_loaded = False
        self._last_update = None
        
        # Statistics
        self._stats = {
            'total_datasets': 0,
            'trading_relevant': 0,
            'categories': {},
            'last_loaded': None,
            'load_time_seconds': 0
        }
    
    async def load_all_async(self, force_reload: bool = False) -> None:
        """
        Load all datasets from Data.gov.
        
        Args:
            force_reload: Force reload even if recently loaded
        """
        if self._is_loaded and not force_reload:
            return
        
        if not force_reload and self._last_update:
            time_since_update = datetime.now() - self._last_update
            if time_since_update.total_seconds() < self.config.dataset_cache_ttl:
                return
        
        logger.info("Loading datasets from Data.gov...")
        start_time = datetime.now()
        
        try:
            await self._load_datasets()
            self._build_indices()
            self._update_statistics()
            
            load_time = (datetime.now() - start_time).total_seconds()
            self._stats['load_time_seconds'] = load_time
            self._stats['last_loaded'] = datetime.now().isoformat()
            
            self._is_loaded = True
            self._last_update = datetime.now()
            
            logger.info(f"Loaded {len(self._datasets)} datasets in {load_time:.2f}s "
                       f"({self._stats['trading_relevant']} trading-relevant)")
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    async def _load_datasets(self) -> None:
        """Load datasets from API with batching and filtering."""
        datasets_loaded = 0
        start = 0
        batch_size = 100
        
        # Clear existing data
        self._datasets.clear()
        self._dataset_index.clear()
        self._category_index.clear()
        
        while True:
            try:
                # Get batch of datasets
                results = await self.api_client.search_datasets(
                    query="*:*",
                    start=start,
                    rows=batch_size,
                    sort="metadata_modified desc",
                    trading_relevant_only=self.config.trading_relevant_only
                )
                
                if not results.results:
                    break
                
                # Process datasets
                for dataset in results.results:
                    if self._should_include_dataset(dataset):
                        self._datasets[dataset.id] = dataset
                        datasets_loaded += 1
                
                # Check if we have enough datasets or reached end
                start += batch_size
                if (datasets_loaded >= 1000 or  # Reasonable limit
                    start >= results.count or
                    len(results.results) < batch_size):
                    break
                
                # Rate limiting - small delay between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to load dataset batch at {start}: {e}")
                break
    
    def _should_include_dataset(self, dataset: DatagovDataset) -> bool:
        """
        Determine if dataset should be included.
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            True if dataset should be included
        """
        # Check category filter
        if self.config.include_categories:
            if dataset.category.value not in self.config.include_categories:
                return False
        
        # Check minimum resources
        data_resources = dataset.data_resources
        if len(data_resources) < self.config.min_resources:
            return False
        
        # Check excluded formats
        if self.config.exclude_formats:
            for resource in data_resources:
                if resource.format.lower() in self.config.exclude_formats:
                    return False
        
        # Trading relevance check
        if self.config.trading_relevant_only:
            relevance = assess_trading_relevance(dataset.dict())
            if relevance < 0.2:  # Minimum threshold
                return False
        
        return True
    
    def _build_indices(self) -> None:
        """Build search indices for fast lookups."""
        self._dataset_index.clear()
        self._category_index.clear()
        
        for dataset_id, dataset in self._datasets.items():
            # Title index
            if dataset.title:
                title_key = dataset.title.lower().strip()
                self._dataset_index[title_key] = dataset_id
            
            # Category index
            category = dataset.category
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(dataset_id)
    
    def _update_statistics(self) -> None:
        """Update internal statistics."""
        self._stats['total_datasets'] = len(self._datasets)
        
        trading_relevant = 0
        categories = {}
        
        for dataset in self._datasets.values():
            # Count trading relevant
            relevance = assess_trading_relevance(dataset.dict())
            if relevance >= 0.3:
                trading_relevant += 1
            
            # Count by category
            category = dataset.category.value
            categories[category] = categories.get(category, 0) + 1
        
        self._stats['trading_relevant'] = trading_relevant
        self._stats['categories'] = categories
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[DatagovDataset]:
        """
        Get dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset or None if not found
        """
        return self._datasets.get(dataset_id)
    
    def search_datasets(self, query: str, limit: int = 10) -> List[DatagovDataset]:
        """
        Search datasets by query.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching datasets
        """
        query_lower = query.lower().strip()
        matches = []
        
        # Exact title match first
        if query_lower in self._dataset_index:
            dataset_id = self._dataset_index[query_lower]
            dataset = self._datasets.get(dataset_id)
            if dataset:
                matches.append(dataset)
        
        # Partial matches
        for dataset in self._datasets.values():
            if len(matches) >= limit:
                break
            
            # Skip if already in exact matches
            if dataset in matches:
                continue
            
            # Check title, description, and tags
            text_to_search = [
                dataset.title or '',
                dataset.notes or '',
                ' '.join(dataset.tags)
            ]
            
            if any(query_lower in text.lower() for text in text_to_search):
                matches.append(dataset)
        
        return matches[:limit]
    
    def get_datasets_by_category(self, category: DatasetCategory) -> List[DatagovDataset]:
        """
        Get datasets by category.
        
        Args:
            category: Dataset category
            
        Returns:
            List of datasets in category
        """
        dataset_ids = self._category_index.get(category, [])
        return [self._datasets[dataset_id] for dataset_id in dataset_ids 
                if dataset_id in self._datasets]
    
    def get_trading_relevant_datasets(self, min_relevance: float = 0.3) -> List[DatagovDataset]:
        """
        Get datasets relevant for trading strategies.
        
        Args:
            min_relevance: Minimum relevance score
            
        Returns:
            List of relevant datasets sorted by relevance
        """
        relevant_datasets = []
        
        for dataset in self._datasets.values():
            relevance = assess_trading_relevance(dataset.dict())
            if relevance >= min_relevance:
                relevant_datasets.append((dataset, relevance))
        
        # Sort by relevance score
        relevant_datasets.sort(key=lambda x: x[1], reverse=True)
        
        return [dataset for dataset, _ in relevant_datasets]
    
    def get_real_time_datasets(self) -> List[DatagovDataset]:
        """Get datasets with real-time or frequent updates."""
        real_time_datasets = []
        
        for dataset in self._datasets.values():
            frequency = dataset.estimated_frequency
            if frequency in [DatasetFrequency.REAL_TIME, DatasetFrequency.DAILY]:
                real_time_datasets.append(dataset)
        
        return real_time_datasets
    
    def get_datasets_by_organization(self, organization_name: str) -> List[DatagovDataset]:
        """
        Get datasets by organization.
        
        Args:
            organization_name: Organization name or ID
            
        Returns:
            List of datasets from organization
        """
        org_name_lower = organization_name.lower()
        matching_datasets = []
        
        for dataset in self._datasets.values():
            if dataset.organization:
                org_names = [
                    dataset.organization.name.lower() if dataset.organization.name else '',
                    dataset.organization.id.lower() if dataset.organization.id else '',
                    dataset.organization.title.lower() if dataset.organization.title else ''
                ]
                
                if any(org_name_lower in org_name for org_name in org_names):
                    matching_datasets.append(dataset)
        
        return matching_datasets
    
    def create_instrument_id(self, dataset: DatagovDataset) -> str:
        """
        Create NautilusTrader instrument ID from dataset.
        
        Args:
            dataset: Dataset to create ID for
            
        Returns:
            Instrument ID string
        """
        # Use sanitized dataset name as base
        base_name = sanitize_dataset_name(dataset.name or dataset.title)
        
        # Add category prefix
        category_prefix = dataset.category.value.upper()[:3]
        
        # Format: CAT_DATASET_NAME.DATAGOV
        return f"{category_prefix}_{base_name}.DATAGOV"
    
    def get_instrument_metadata(self, dataset: DatagovDataset) -> Dict[str, Any]:
        """
        Get instrument metadata for NautilusTrader.
        
        Args:
            dataset: Dataset to get metadata for
            
        Returns:
            Instrument metadata dictionary
        """
        return {
            'dataset_id': dataset.id,
            'title': dataset.title,
            'description': dataset.notes,
            'category': dataset.category.value,
            'organization': dataset.organization.name if dataset.organization else None,
            'tags': dataset.tags,
            'resource_count': len(dataset.resources),
            'data_resource_count': len(dataset.data_resources),
            'api_resource_count': len(dataset.api_resources),
            'estimated_frequency': dataset.estimated_frequency.value,
            'trading_relevance': assess_trading_relevance(dataset.dict()),
            'created': dataset.metadata_created,
            'modified': dataset.metadata_modified,
            'url': dataset.url
        }
    
    async def check_for_updates(self) -> List[str]:
        """
        Check for dataset updates.
        
        Returns:
            List of updated dataset IDs
        """
        if not self.config.monitor_updates:
            return []
        
        updated_ids = []
        
        try:
            # Get recently modified datasets
            cutoff_date = datetime.now() - timedelta(hours=24)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            results = await self.api_client.search_datasets(
                query=f"metadata_modified:[{cutoff_str} TO *]",
                rows=100,
                sort="metadata_modified desc"
            )
            
            for dataset in results.results:
                if dataset.id in self._datasets:
                    # Check if actually updated
                    existing = self._datasets[dataset.id]
                    if existing.metadata_modified != dataset.metadata_modified:
                        updated_ids.append(dataset.id)
                        self._datasets[dataset.id] = dataset
            
            if updated_ids:
                self._build_indices()
                self._update_statistics()
                logger.info(f"Found {len(updated_ids)} updated datasets")
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
        
        return updated_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            **self._stats,
            'is_loaded': self._is_loaded,
            'datasets_in_memory': len(self._datasets),
            'categories_available': len(self._category_index),
            'index_size': len(self._dataset_index)
        }
    
    def get_dataset_count(self) -> int:
        """Get total dataset count."""
        return len(self._datasets)
    
    @property
    def is_loaded(self) -> bool:
        """Check if datasets are loaded."""
        return self._is_loaded