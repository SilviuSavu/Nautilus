"""
Data.gov Integration Service
============================

Professional-grade integration with Data.gov CKAN API for accessing
346,000+ federal datasets relevant to algorithmic trading strategies.

Features:
- Dataset discovery and categorization
- Trading relevance scoring
- Real-time update monitoring
- Cached data for performance
- Rate limiting and error handling
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

from datagov_connector import (
    DatagovAPIClient,
    create_default_config,
    DatagovInstrumentProvider,
    create_instrument_config,
    DatagovDataset,
    DatasetCategory,
    DatasetFrequency
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetFactor:
    """Calculated factor from dataset analysis."""
    name: str
    value: float
    category: str
    dataset_id: str
    calculation_date: str
    confidence: float = 1.0


class DatagovIntegrationService:
    """
    Data.gov API Integration for institutional-grade dataset access.
    
    Provides access to 346,000+ federal datasets organized by categories:
    - Economic indicators and financial data
    - Agricultural commodity reports
    - Energy production and pricing data
    - Transportation and infrastructure metrics
    - Environmental and climate data
    """
    
    def __init__(self):
        self.api_key = os.environ.get('DATAGOV_API_KEY')
        
        # Initialize API client
        if self.api_key:
            self.config = create_default_config(self.api_key)
            self.api_client = DatagovAPIClient(self.config)
        else:
            self.config = None
            self.api_client = None
            logger.warning("DATAGOV_API_KEY not set - service will have limited functionality")
        
        # Initialize instrument provider
        if self.api_client:
            instrument_config = create_instrument_config()
            self.instrument_provider = DatagovInstrumentProvider(self.api_client, instrument_config)
        else:
            self.instrument_provider = None
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'errors': 0,
            'last_request_time': None,
            'datasets_loaded': 0,
            'trading_relevant_datasets': 0
        }
        
        logger.info(f"Data.gov Integration initialized - API key configured: {bool(self.api_key)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Data.gov integration."""
        health_status = {
            "service": "Data.gov Integration",
            "status": "unknown",
            "api_key_configured": bool(self.api_key),
            "api_accessible": False,
            "datasets_loaded": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if not self.api_key:
            health_status.update({
                "status": "not_configured",
                "error_message": "DATAGOV_API_KEY environment variable not set"
            })
            return health_status
        
        try:
            # Test API connectivity
            async with self.api_client:
                api_healthy = await self.api_client.health_check()
                health_status["api_accessible"] = api_healthy
                
                if api_healthy:
                    # Get basic stats
                    if self.instrument_provider and self.instrument_provider.is_loaded:
                        stats = self.instrument_provider.get_statistics()
                        health_status.update({
                            "status": "operational",
                            "datasets_loaded": stats['total_datasets'],
                            "trading_relevant": stats['trading_relevant'],
                            "categories": stats['categories'],
                            "last_loaded": stats['last_loaded']
                        })
                    else:
                        health_status.update({
                            "status": "operational",
                            "message": "API accessible but datasets not loaded"
                        })
                else:
                    health_status.update({
                        "status": "degraded",
                        "error_message": "API not accessible"
                    })
                    
        except Exception as e:
            health_status.update({
                "status": "error",
                "error_message": str(e)
            })
        
        return health_status
    
    async def load_datasets(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load datasets from Data.gov.
        
        Args:
            force_reload: Force reload even if recently loaded
            
        Returns:
            Load summary
        """
        if not self.instrument_provider:
            raise ValueError("Service not properly initialized - check API key")
        
        try:
            start_time = datetime.now()
            
            await self.instrument_provider.load_all_async(force_reload)
            
            load_time = (datetime.now() - start_time).total_seconds()
            stats = self.instrument_provider.get_statistics()
            
            self._stats.update({
                'datasets_loaded': stats['total_datasets'],
                'trading_relevant_datasets': stats['trading_relevant']
            })
            
            return {
                'success': True,
                'datasets_loaded': stats['total_datasets'],
                'trading_relevant': stats['trading_relevant'],
                'categories': stats['categories'],
                'load_time_seconds': load_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    async def search_datasets(
        self,
        query: str = "*:*",
        category: Optional[str] = None,
        organization: Optional[str] = None,
        limit: int = 20,
        trading_relevant_only: bool = True
    ) -> Dict[str, Any]:
        """
        Search for datasets.
        
        Args:
            query: Search query
            category: Filter by category
            organization: Filter by organization  
            limit: Maximum results
            trading_relevant_only: Only return trading-relevant datasets
            
        Returns:
            Search results
        """
        if not self.api_client:
            raise ValueError("API client not initialized - check API key")
        
        try:
            async with self.api_client:
                results = await self.api_client.search_datasets(
                    query=query,
                    organization=organization,
                    rows=limit,
                    trading_relevant_only=trading_relevant_only
                )
                
                # Filter by category if specified
                filtered_results = results.results
                if category:
                    try:
                        cat_enum = DatasetCategory(category.lower())
                        filtered_results = [d for d in results.results if d.category == cat_enum]
                    except ValueError:
                        logger.warning(f"Invalid category: {category}")
                
                return {
                    'success': True,
                    'query': query,
                    'total_count': results.count,
                    'returned_count': len(filtered_results),
                    'datasets': [self._format_dataset_summary(d) for d in filtered_results],
                    'facets': results.facets,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Dataset search failed: {e}")
            raise
    
    async def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get detailed dataset information.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset details
        """
        if not self.api_client:
            raise ValueError("API client not initialized")
        
        try:
            async with self.api_client:
                dataset = await self.api_client.get_dataset(dataset_id)
                
                if not dataset:
                    return {
                        'success': False,
                        'dataset_id': dataset_id,
                        'error': 'Dataset not found'
                    }
                
                return {
                    'success': True,
                    'dataset': self._format_dataset_details(dataset),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            raise
    
    async def get_trading_relevant_datasets(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get datasets most relevant for trading strategies.
        
        Args:
            limit: Maximum number of datasets
            
        Returns:
            Trading-relevant datasets
        """
        if not self.instrument_provider:
            raise ValueError("Service not initialized")
        
        try:
            # Ensure datasets are loaded
            if not self.instrument_provider.is_loaded:
                await self.instrument_provider.load_all_async()
            
            relevant_datasets = self.instrument_provider.get_trading_relevant_datasets()
            
            return {
                'success': True,
                'count': len(relevant_datasets),
                'datasets': [self._format_dataset_summary(d) for d in relevant_datasets[:limit]],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading-relevant datasets: {e}")
            raise
    
    async def get_datasets_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get datasets by category.
        
        Args:
            category: Dataset category
            
        Returns:
            Datasets in category
        """
        if not self.instrument_provider:
            raise ValueError("Service not initialized")
        
        try:
            # Ensure datasets are loaded
            if not self.instrument_provider.is_loaded:
                await self.instrument_provider.load_all_async()
            
            try:
                cat_enum = DatasetCategory(category.lower())
            except ValueError:
                return {
                    'success': False,
                    'error': f'Invalid category: {category}',
                    'valid_categories': [c.value for c in DatasetCategory]
                }
            
            datasets = self.instrument_provider.get_datasets_by_category(cat_enum)
            
            return {
                'success': True,
                'category': category,
                'count': len(datasets),
                'datasets': [self._format_dataset_summary(d) for d in datasets],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get datasets for category {category}: {e}")
            raise
    
    async def get_organizations(self) -> Dict[str, Any]:
        """Get list of government organizations."""
        if not self.api_client:
            raise ValueError("API client not initialized")
        
        try:
            async with self.api_client:
                organizations = await self.api_client.list_organizations()
                
                return {
                    'success': True,
                    'count': len(organizations),
                    'organizations': [
                        {
                            'id': org.id,
                            'name': org.name,
                            'title': org.title,
                            'description': org.description
                        }
                        for org in organizations
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get organizations: {e}")
            raise
    
    async def get_real_time_datasets(self) -> Dict[str, Any]:
        """Get datasets with real-time or frequent updates."""
        if not self.instrument_provider:
            raise ValueError("Service not initialized")
        
        try:
            # Ensure datasets are loaded
            if not self.instrument_provider.is_loaded:
                await self.instrument_provider.load_all_async()
            
            real_time_datasets = self.instrument_provider.get_real_time_datasets()
            
            return {
                'success': True,
                'count': len(real_time_datasets),
                'datasets': [self._format_dataset_summary(d) for d in real_time_datasets],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time datasets: {e}")
            raise
    
    def _format_dataset_summary(self, dataset: DatagovDataset) -> Dict[str, Any]:
        """Format dataset for API response."""
        return {
            'id': dataset.id,
            'name': dataset.name,
            'title': dataset.title,
            'description': dataset.notes,
            'category': dataset.category.value,
            'organization': {
                'id': dataset.organization.id if dataset.organization else None,
                'name': dataset.organization.name if dataset.organization else None,
                'title': dataset.organization.title if dataset.organization else None
            },
            'tags': dataset.tags,
            'resource_count': len(dataset.resources),
            'data_resources': len(dataset.data_resources),
            'api_resources': len(dataset.api_resources),
            'estimated_frequency': dataset.estimated_frequency.value,
            'trading_relevant': dataset.is_trading_relevant,
            'created': dataset.metadata_created,
            'modified': dataset.metadata_modified,
            'url': dataset.url
        }
    
    def _format_dataset_details(self, dataset: DatagovDataset) -> Dict[str, Any]:
        """Format detailed dataset information."""
        summary = self._format_dataset_summary(dataset)
        
        # Add detailed resource information
        summary['resources'] = [
            {
                'id': resource.id,
                'name': resource.name,
                'description': resource.description,
                'url': resource.url,
                'format': resource.format,
                'size': resource.size,
                'created': resource.created,
                'modified': resource.last_modified,
                'is_data_resource': resource.is_data_resource
            }
            for resource in dataset.resources
        ]
        
        # Add extra metadata
        summary['extras'] = dataset.extras
        
        return summary
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        stats = dict(self._stats)
        
        # Add API client stats
        if self.api_client:
            api_stats = self.api_client.get_stats()
            stats.update({
                'api_requests': api_stats['total_requests'],
                'api_cache_hits': api_stats['cache_hits'],
                'api_rate_limited': api_stats['rate_limited'],
                'api_errors': api_stats['errors']
            })
        
        # Add instrument provider stats
        if self.instrument_provider and self.instrument_provider.is_loaded:
            provider_stats = self.instrument_provider.get_statistics()
            stats.update(provider_stats)
        
        return {
            'service': 'Data.gov Integration',
            'status': 'operational' if self.api_key else 'not_configured',
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close API client connections."""
        if self.api_client:
            await self.api_client.close()


# Global instance
datagov_integration = DatagovIntegrationService()