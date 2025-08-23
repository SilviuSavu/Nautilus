"""
Data.gov Utilities
==================

Utility functions for Data.gov integration, following EDGAR patterns.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def validate_dataset_id(dataset_id: str) -> str:
    """
    Validate and normalize Data.gov dataset ID.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Normalized dataset ID
        
    Raises:
        ValueError: If dataset ID is invalid
    """
    if not dataset_id:
        raise ValueError("Dataset ID cannot be empty")
    
    # Dataset IDs are typically lowercase with hyphens
    normalized = dataset_id.strip().lower()
    
    # Basic validation - should be alphanumeric with hyphens and underscores
    if not re.match(r'^[a-z0-9\-_]+$', normalized):
        raise ValueError(f"Invalid dataset ID format: {dataset_id}")
    
    return normalized


def validate_organization(org_name: str) -> str:
    """
    Validate and normalize organization name.
    
    Args:
        org_name: Organization name or ID
        
    Returns:
        Normalized organization name
        
    Raises:
        ValueError: If organization name is invalid
    """
    if not org_name:
        raise ValueError("Organization name cannot be empty")
    
    # Normalize to lowercase with hyphens
    normalized = org_name.strip().lower().replace(' ', '-')
    
    # Remove special characters except hyphens and underscores
    normalized = re.sub(r'[^a-z0-9\-_]', '', normalized)
    
    if not normalized:
        raise ValueError(f"Invalid organization name: {org_name}")
    
    return normalized


def validate_api_key(api_key: str) -> bool:
    """
    Validate Data.gov API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid format
    """
    if not api_key:
        return False
    
    # API keys are typically UUIDs or alphanumeric strings
    # Allow for various formats used by different providers
    return len(api_key) >= 16 and re.match(r'^[a-zA-Z0-9\-_]+$', api_key)


def format_dataset_title(title: str) -> str:
    """
    Format dataset title for display.
    
    Args:
        title: Raw dataset title
        
    Returns:
        Formatted title
    """
    if not title:
        return "Unknown Dataset"
    
    # Clean up common formatting issues
    formatted = title.strip()
    
    # Remove excessive whitespace
    formatted = re.sub(r'\s+', ' ', formatted)
    
    # Capitalize first letter if not already
    if formatted and formatted[0].islower():
        formatted = formatted[0].upper() + formatted[1:]
    
    return formatted


def extract_trading_keywords(text: str) -> List[str]:
    """
    Extract trading-relevant keywords from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of relevant keywords found
    """
    if not text:
        return []
    
    text_lower = text.lower()
    
    trading_keywords = {
        # Economic indicators
        'gdp', 'inflation', 'unemployment', 'employment', 'wages', 'cpi', 'ppi',
        'interest rates', 'federal funds', 'treasury', 'bonds', 'yield curve',
        
        # Financial markets
        'stocks', 'equities', 'bonds', 'commodities', 'currencies', 'forex',
        'derivatives', 'futures', 'options', 'securities', 'banking', 'credit',
        
        # Commodities
        'oil', 'gas', 'gold', 'silver', 'copper', 'wheat', 'corn', 'soybeans',
        'agriculture', 'energy', 'petroleum', 'natural gas',
        
        # Market data
        'prices', 'trading', 'volume', 'market', 'exchange', 'index',
        'performance', 'volatility', 'returns'
    }
    
    found_keywords = []
    for keyword in trading_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def assess_trading_relevance(dataset: Dict[str, Any]) -> float:
    """
    Assess how relevant a dataset is for trading strategies.
    
    Args:
        dataset: Dataset metadata dictionary
        
    Returns:
        Relevance score (0.0 - 1.0)
    """
    score = 0.0
    
    title = dataset.get('title', '')
    description = dataset.get('notes', '') or dataset.get('description', '')
    tags = [tag.get('name', '') if isinstance(tag, dict) else str(tag) 
            for tag in dataset.get('tags', [])]
    
    # Check title for trading keywords
    title_keywords = extract_trading_keywords(title)
    score += len(title_keywords) * 0.2
    
    # Check description
    desc_keywords = extract_trading_keywords(description)
    score += len(desc_keywords) * 0.1
    
    # Check tags
    tag_text = ' '.join(tags)
    tag_keywords = extract_trading_keywords(tag_text)
    score += len(tag_keywords) * 0.1
    
    # Organization bonus
    org = dataset.get('organization', {})
    if isinstance(org, dict):
        org_name = org.get('name', '').lower()
        if any(keyword in org_name for keyword in [
            'fed', 'treasury', 'commerce', 'labor', 'energy', 'agriculture'
        ]):
            score += 0.3
    
    # Resource format bonus (APIs and structured data preferred)
    resources = dataset.get('resources', [])
    for resource in resources:
        format_type = resource.get('format', '').lower()
        if format_type in ['json', 'csv', 'xml', 'api']:
            score += 0.1
        elif format_type in ['pdf', 'html']:
            score -= 0.1
    
    # Normalize score
    return min(1.0, max(0.0, score))


def parse_ckan_datetime(date_str: str) -> Optional[datetime]:
    """
    Parse CKAN datetime string to datetime object.
    
    Args:
        date_str: CKAN date string
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_str:
        return None
    
    # Common CKAN date formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def build_search_query(
    keywords: Optional[List[str]] = None,
    organizations: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    categories: Optional[List[str]] = None
) -> str:
    """
    Build CKAN search query string.
    
    Args:
        keywords: Search keywords
        organizations: Organization filters
        tags: Tag filters
        categories: Category filters
        
    Returns:
        CKAN query string
    """
    query_parts = []
    
    if keywords:
        keyword_query = ' OR '.join(f'"{keyword}"' for keyword in keywords)
        query_parts.append(f"({keyword_query})")
    
    if organizations:
        org_query = ' OR '.join(f'organization:"{org}"' for org in organizations)
        query_parts.append(f"({org_query})")
    
    if tags:
        tag_query = ' OR '.join(f'tags:"{tag}"' for tag in tags)
        query_parts.append(f"({tag_query})")
    
    if categories:
        # Categories might be in groups or tags
        cat_query = ' OR '.join(f'(tags:"{cat}" OR groups:"{cat}")' for cat in categories)
        query_parts.append(f"({cat_query})")
    
    return ' AND '.join(query_parts) if query_parts else '*:*'


def format_file_size(size_str: Optional[str]) -> Optional[str]:
    """
    Format file size string for display.
    
    Args:
        size_str: Size string from API
        
    Returns:
        Formatted size string or None
    """
    if not size_str:
        return None
    
    try:
        # Try to parse as integer (bytes)
        size_bytes = int(size_str)
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} PB"
        
    except ValueError:
        # Return as-is if can't parse
        return size_str


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_dataset_name(name: str) -> str:
    """
    Sanitize dataset name for use as identifier.
    
    Args:
        name: Raw dataset name
        
    Returns:
        Sanitized name suitable for use as identifier
    """
    if not name:
        return "unknown_dataset"
    
    # Convert to lowercase and replace spaces/special chars with underscores
    sanitized = re.sub(r'[^a-z0-9]', '_', name.lower())
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure it's not empty and not too long
    if not sanitized:
        sanitized = "unknown_dataset"
    elif len(sanitized) > 64:
        sanitized = sanitized[:64].rstrip('_')
    
    return sanitized