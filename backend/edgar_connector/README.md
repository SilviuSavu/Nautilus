# EDGAR API Connector for NautilusTrader

A comprehensive data connector that bridges the SEC's EDGAR API with NautilusTrader's unified architecture, providing structured access to SEC filings and financial data for trading algorithms and research.

## Overview

The EDGAR API connector enables NautilusTrader strategies to access:
- SEC company filings (10-K, 10-Q, 8-K, etc.)
- Financial facts and metrics from XBRL data
- Company entity mappings (CIK ↔ Ticker)
- Real-time filing notifications
- Historical submission data

## Architecture

```
SEC EDGAR API ← EDGARAPIClient ← EDGARDataClient ← NautilusTrader MessageBus
                      ↓
              EDGARInstrumentProvider ← Strategy Components
```

### Key Components

- **EDGARAPIClient**: HTTP client with rate limiting and caching
- **EDGARDataClient**: NautilusTrader LiveDataClient implementation
- **EDGARInstrumentProvider**: SEC entity management and ticker resolution
- **Custom Data Types**: FilingData, CompanyFacts, SECFiling
- **Utilities**: XBRL parsing, data normalization, caching

## Installation

### Dependencies

Add to `requirements.txt`:
```
edgar-sec>=0.5.0
sec-api>=1.1.0
httpx>=0.25.2
```

Install:
```bash
pip install edgar-sec sec-api httpx
```

### Configuration

```python
from edgar_connector.config import create_default_config

config = create_default_config(
    user_agent="YourCompany contact@yourcompany.com",  # Required by SEC
    rate_limit_requests_per_second=5.0,  # Max 10/sec allowed
    cache_ttl_seconds=1800,  # 30 minutes
    enable_cache=True
)
```

## Quick Start

### Basic API Usage

```python
import asyncio
from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import create_default_config

async def main():
    config = create_default_config(
        user_agent="MyApp contact@example.com"
    )
    
    async with EDGARAPIClient(config) as client:
        # Health check
        healthy = await client.health_check()
        print(f"API Status: {'OK' if healthy else 'Error'}")
        
        # Get Apple's financial facts
        facts = await client.get_company_facts("0000320193")
        print(f"Company: {facts['entityName']}")
        
        # Resolve ticker to CIK
        cik = await client.resolve_ticker_to_cik("AAPL")
        print(f"AAPL CIK: {cik}")

asyncio.run(main())
```

### Instrument Provider Usage

```python
from edgar_connector.instrument_provider import EDGARInstrumentProvider
from edgar_connector.config import EDGARInstrumentConfig

# Setup
instrument_config = EDGARInstrumentConfig(
    update_entities_on_startup=True,
    ticker_cache_ttl=3600
)

provider = EDGARInstrumentProvider(api_client, instrument_config)
await provider.load_all_async()

# Usage
entity = provider.get_entity_by_ticker("AAPL")
cik = provider.resolve_ticker_to_cik("GOOGL")
results = provider.search_entities("Apple")

print(f"Loaded {provider.get_entity_count()} companies")
```

### Data Client Integration

```python
from nautilus_trader.common.component import LiveClock
from nautilus_trader.model.identifiers import ClientId
from edgar_connector.data_client import EDGARDataClient

# Create data client
client_id = ClientId("EDGAR-001")
clock = LiveClock()

data_client = EDGARDataClient(
    client_id=client_id,
    config=edgar_config,
    data_config=data_client_config,
    clock=clock,
    instrument_provider=provider
)

# Connect and use
await data_client._connect()

# Request company facts
facts = await data_client.request_company_facts("0000320193")
print(f"Revenue: {facts.facts.get('revenue')}")

# Request filings
filings = await data_client.request_company_filings("0000320193")
for filing in filings:
    print(f"{filing.filing_type}: {filing.filing_date}")
```

## FastAPI Integration

The connector includes FastAPI routes for REST API access:

```python
# In main.py
from edgar_routes import router as edgar_router
app.include_router(edgar_router)
```

### API Endpoints

#### Health Check
```
GET /api/v1/edgar/health
```

#### Company Search
```
GET /api/v1/edgar/companies/search?q=Apple&limit=10
```

#### Ticker Resolution
```
GET /api/v1/edgar/ticker/AAPL/resolve
```

#### Company Facts
```
GET /api/v1/edgar/companies/0000320193/facts
GET /api/v1/edgar/ticker/AAPL/facts
```

#### Company Filings
```
GET /api/v1/edgar/companies/0000320193/filings?form_types=10-K,10-Q&days_back=365
GET /api/v1/edgar/ticker/AAPL/filings?days_back=90
```

#### Statistics
```
GET /api/v1/edgar/statistics
```

### Example API Responses

**Company Facts**:
```json
{
  "cik": "0000320193",
  "company_name": "Apple Inc.",
  "key_metrics": [
    {
      "metric": "Revenue",
      "value": "$394.33B",
      "raw_value": 394328000000.0,
      "period_end": "2023-09-30"
    }
  ],
  "total_facts": 245
}
```

**Company Filings**:
```json
[
  {
    "form_type": "10-K",
    "filing_date": "2023-11-03",
    "accession_number": "0000320193-23-000123",
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "days_ago": 45
  }
]
```

## Data Types

### Filing Types
```python
from edgar_connector.data_types import FilingType

FilingType.FORM_10K    # Annual report
FilingType.FORM_10Q    # Quarterly report
FilingType.FORM_8K     # Current report
FilingType.FORM_DEF14A # Proxy statement
# ... and more
```

### Custom Data Objects

**FilingData**: NautilusTrader CustomData for SEC filings
**CompanyFacts**: XBRL financial data
**SECEntity**: Company entity information
**SECFiling**: Structured filing information

## Configuration Options

### EDGARConfig
- `user_agent`: Required contact email
- `rate_limit_requests_per_second`: Max 10/sec (SEC limit)
- `cache_ttl_seconds`: Cache duration
- `max_retries`: Retry attempts
- `request_timeout`: HTTP timeout

### EDGARDataClientConfig
- `auto_subscribe_filings`: Auto-monitor new filings
- `subscription_check_interval`: Check frequency
- `parse_financial_data`: Parse XBRL data
- `max_filing_age_days`: Filter old filings

## Rate Limiting & Compliance

The connector automatically enforces SEC rate limits:
- Maximum 10 requests per second
- Exponential backoff on rate limit errors
- Proper User-Agent header with contact info
- Respectful caching to reduce load

## Error Handling

```python
try:
    facts = await client.get_company_facts("invalid_cik")
except Exception as e:
    logger.error(f"API error: {e}")
```

All methods include comprehensive error handling with logging.

## Testing

### Unit Tests
```bash
cd backend
pytest tests/edgar_connector/test_*.py -v
```

### Integration Tests
```bash
cd backend
python test_edgar_integration.py
```

### Real API Tests (Slow)
```bash
pytest tests/edgar_connector/ -v -m "integration and slow"
```

## Performance Considerations

- **Caching**: Enabled by default with 30-minute TTL
- **Rate Limiting**: Automatic compliance with SEC limits
- **Concurrent Requests**: Configurable via `concurrent_requests`
- **Memory Usage**: LRU cache for frequently accessed data

## Security & Compliance

- ✅ User-Agent header with contact information
- ✅ Rate limiting compliance (≤10 req/sec)
- ✅ No API keys stored in code
- ✅ HTTPS-only requests
- ✅ Input validation and sanitization

## Production Deployment

### Environment Variables
```bash
EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND=5.0
EDGAR_CACHE_TTL_SECONDS=3600
EDGAR_MAX_RETRIES=3
EDGAR_USER_AGENT="YourApp support@yourcompany.com"
```

### Monitoring
```python
# Check service health
response = requests.get("http://localhost:8000/api/v1/edgar/health")
status = response.json()
```

## Troubleshooting

### Common Issues

**Rate Limiting**: 
- Reduce `rate_limit_requests_per_second`
- Increase cache TTL
- Check User-Agent header format

**CIK Resolution**:
- Use 10-digit zero-padded format
- Verify ticker exists in SEC database
- Check entity provider is loaded

**Missing Data**:
- Verify CIK is valid
- Check filing date filters
- Ensure company has filed required forms

### Debug Logging
```python
import logging
logging.getLogger("edgar_connector").setLevel(logging.DEBUG)
```

## Contributing

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Verify SEC API compliance
5. Test with real data

## License

This connector follows the same license as the main NautilusTrader project.

## Support

For issues and questions:
- Check the troubleshooting section
- Review test files for examples  
- Create an issue in the main repository

## Roadmap

- [ ] WebSocket support for real-time filings
- [ ] Advanced XBRL parsing
- [ ] Fundamental analysis indicators
- [ ] ESG data integration
- [ ] Options and derivatives filings