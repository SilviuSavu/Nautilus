"""
Data.gov Data Types and Models
==============================

Pydantic models and enums for Data.gov dataset structures.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator


class DatasetCategory(Enum):
    """Dataset categories for factor modeling and trading analysis."""
    ECONOMIC = "economic"
    AGRICULTURAL = "agricultural"
    ENERGY = "energy"
    DEMOGRAPHIC = "demographic"
    TRANSPORTATION = "transportation"
    HEALTH = "health"
    FINANCIAL = "financial"
    ENVIRONMENTAL = "environmental"
    EDUCATION = "education"
    SAFETY = "safety"
    OTHER = "other"


class DatasetFrequency(Enum):
    """Dataset update frequencies."""
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class ResourceFormat(Enum):
    """Supported resource formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    API = "api"
    RSS = "rss"
    PDF = "pdf"
    XLS = "xls"
    XLSX = "xlsx"
    ZIP = "zip"
    OTHER = "other"


class DatasetResource(BaseModel):
    """Individual dataset resource (file, API endpoint, etc.)."""
    id: str = Field(description="Resource ID")
    name: str = Field(description="Resource name")
    description: Optional[str] = Field(None, description="Resource description")
    url: str = Field(description="Resource URL")
    format: str = Field(description="Resource format (csv, json, api, etc.)")
    size: Optional[str] = Field(None, description="Resource size")
    created: Optional[str] = Field(None, description="Creation date")
    last_modified: Optional[str] = Field(None, description="Last modified date")
    mimetype: Optional[str] = Field(None, description="MIME type")
    
    @property
    def format_enum(self) -> ResourceFormat:
        """Get format as enum."""
        try:
            return ResourceFormat(self.format.lower())
        except ValueError:
            return ResourceFormat.OTHER
            
    @property
    def is_data_resource(self) -> bool:
        """Check if resource contains data."""
        data_formats = {ResourceFormat.JSON, ResourceFormat.CSV, ResourceFormat.XML, ResourceFormat.API}
        return self.format_enum in data_formats


class OrganizationInfo(BaseModel):
    """Government organization information."""
    id: str = Field(description="Organization ID")
    name: str = Field(description="Organization name")
    title: str = Field(description="Organization display title")
    description: Optional[str] = Field(None, description="Organization description")
    image_url: Optional[str] = Field(None, description="Organization logo URL")
    created: Optional[str] = Field(None, description="Creation date")


class DatagovDataset(BaseModel):
    """Complete Data.gov dataset representation."""
    id: str = Field(description="Dataset ID")
    name: str = Field(description="Dataset machine name")
    title: str = Field(description="Dataset display title")
    notes: Optional[str] = Field(None, description="Dataset description")
    url: Optional[str] = Field(None, description="Dataset page URL")
    
    # Metadata
    author: Optional[str] = Field(None, description="Dataset author")
    author_email: Optional[str] = Field(None, description="Author email")
    maintainer: Optional[str] = Field(None, description="Dataset maintainer")
    maintainer_email: Optional[str] = Field(None, description="Maintainer email")
    
    # Organization
    organization: Optional[OrganizationInfo] = Field(None, description="Publishing organization")
    
    # Resources
    resources: List[DatasetResource] = Field(default_factory=list, description="Dataset resources")
    
    # Classification
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    groups: List[str] = Field(default_factory=list, description="Dataset groups/categories")
    
    # Temporal information
    metadata_created: Optional[str] = Field(None, description="Metadata creation date")
    metadata_modified: Optional[str] = Field(None, description="Metadata last modified")
    
    # Additional metadata
    extras: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('resources', pre=True)
    def parse_resources(cls, v):
        """Parse resources from dict or list format."""
        if isinstance(v, list):
            return [DatasetResource(**r) if isinstance(r, dict) else r for r in v]
        return v
        
    @validator('organization', pre=True)
    def parse_organization(cls, v):
        """Parse organization from dict format."""
        if isinstance(v, dict):
            return OrganizationInfo(**v)
        return v
    
    @validator('tags', pre=True)
    def parse_tags(cls, v):
        """Parse tags from CKAN format."""
        if isinstance(v, list):
            parsed_tags = []
            for tag in v:
                if isinstance(tag, dict):
                    # CKAN format: {"display_name": "tag_name", ...}
                    tag_name = tag.get('display_name') or tag.get('name', str(tag))
                    parsed_tags.append(tag_name)
                else:
                    parsed_tags.append(str(tag))
            return parsed_tags
        return v
    
    @validator('extras', pre=True)
    def parse_extras(cls, v):
        """Parse extras from CKAN list format to dict."""
        if isinstance(v, list):
            extras_dict = {}
            for item in v:
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    extras_dict[item['key']] = item['value']
            return extras_dict
        return v if isinstance(v, dict) else {}
    
    @property
    def category(self) -> DatasetCategory:
        """Determine dataset category based on tags and organization."""
        # Economic indicators
        economic_keywords = {'economic', 'finance', 'budget', 'spending', 'revenue', 'gdp', 'employment', 'wages'}
        if any(keyword in self.title.lower() or keyword in (self.notes or '').lower() 
               or any(keyword in tag.lower() for tag in self.tags) for keyword in economic_keywords):
            return DatasetCategory.ECONOMIC
            
        # Agricultural data
        agricultural_keywords = {'agriculture', 'farming', 'crop', 'livestock', 'food', 'usda'}
        if any(keyword in self.title.lower() or keyword in (self.notes or '').lower() 
               or any(keyword in tag.lower() for tag in self.tags) for keyword in agricultural_keywords):
            return DatasetCategory.AGRICULTURAL
            
        # Energy data
        energy_keywords = {'energy', 'power', 'electricity', 'oil', 'gas', 'renewable', 'coal'}
        if any(keyword in self.title.lower() or keyword in (self.notes or '').lower() 
               or any(keyword in tag.lower() for tag in self.tags) for keyword in energy_keywords):
            return DatasetCategory.ENERGY
            
        # Transportation
        transport_keywords = {'transportation', 'traffic', 'transit', 'highway', 'aviation', 'shipping'}
        if any(keyword in self.title.lower() or keyword in (self.notes or '').lower() 
               or any(keyword in tag.lower() for tag in self.tags) for keyword in transport_keywords):
            return DatasetCategory.TRANSPORTATION
            
        # Environmental
        env_keywords = {'environment', 'climate', 'weather', 'pollution', 'air quality', 'water'}
        if any(keyword in self.title.lower() or keyword in (self.notes or '').lower() 
               or any(keyword in tag.lower() for tag in self.tags) for keyword in env_keywords):
            return DatasetCategory.ENVIRONMENTAL
            
        return DatasetCategory.OTHER
    
    @property
    def data_resources(self) -> List[DatasetResource]:
        """Get only data-containing resources."""
        return [r for r in self.resources if r.is_data_resource]
    
    @property
    def api_resources(self) -> List[DatasetResource]:
        """Get API endpoints."""
        return [r for r in self.resources if r.format_enum == ResourceFormat.API]
        
    @property
    def is_trading_relevant(self) -> bool:
        """Check if dataset is relevant for trading strategies."""
        trading_categories = {
            DatasetCategory.ECONOMIC, 
            DatasetCategory.FINANCIAL, 
            DatasetCategory.ENERGY,
            DatasetCategory.AGRICULTURAL
        }
        return self.category in trading_categories
        
    @property
    def estimated_frequency(self) -> DatasetFrequency:
        """Estimate update frequency from metadata."""
        title_lower = self.title.lower()
        notes_lower = (self.notes or '').lower()
        
        if any(keyword in title_lower or keyword in notes_lower 
               for keyword in ['real-time', 'live', 'streaming']):
            return DatasetFrequency.REAL_TIME
        elif any(keyword in title_lower or keyword in notes_lower 
                 for keyword in ['daily', 'day']):
            return DatasetFrequency.DAILY
        elif any(keyword in title_lower or keyword in notes_lower 
                 for keyword in ['weekly', 'week']):
            return DatasetFrequency.WEEKLY
        elif any(keyword in title_lower or keyword in notes_lower 
                 for keyword in ['monthly', 'month']):
            return DatasetFrequency.MONTHLY
        elif any(keyword in title_lower or keyword in notes_lower 
                 for keyword in ['quarterly', 'quarter']):
            return DatasetFrequency.QUARTERLY
        elif any(keyword in title_lower or keyword in notes_lower 
                 for keyword in ['annual', 'year']):
            return DatasetFrequency.ANNUALLY
            
        return DatasetFrequency.UNKNOWN


class DatasetSearchResult(BaseModel):
    """Search result container."""
    count: int = Field(description="Total number of results")
    results: List[DatagovDataset] = Field(description="Dataset results")
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Search facets")
    
    
class HealthStatus(BaseModel):
    """Data.gov service health status."""
    status: str = Field(description="Overall status")
    timestamp: str = Field(description="Check timestamp")
    api_healthy: bool = Field(description="API connectivity")
    datasets_available: int = Field(description="Number of available datasets")
    cache_enabled: bool = Field(description="Cache status")
    response_time_ms: Optional[float] = Field(None, description="API response time")


class DatagovError(Exception):
    """Data.gov API error."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data