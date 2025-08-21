# FRED API Integration for NautilusTrader

This document provides comprehensive documentation for the Federal Reserve Economic Data (FRED) API integration with NautilusTrader.

## Overview

The FRED adapter provides access to over 800,000 U.S. and international economic time series from 89 sources through the Federal Reserve Bank of St. Louis API. This integration enables quantitative trading strategies to incorporate macroeconomic factors into their decision-making processes.

## Features

- **Comprehensive Data Access**: 800,000+ economic time series
- **Custom Data Types**: `EconomicData` type for economic indicators
- **Intelligent Caching**: Reduces API calls and improves performance
- **Rate Limiting**: Respects FRED API constraints
- **Robust Error Handling**: Graceful handling of API failures
- **Historical Data**: Access to historical economic data for backtesting
- **Real-time Updates**: Periodic checks for new economic data releases

## Installation and Setup

### 1. API Key

Get a free FRED API key from the Federal Reserve Bank of St. Louis:
https://fred.stlouisfed.org/docs/api/api_key.html

### 2. Environment Configuration

Set your API key as an environment variable:
```bash
export FRED_API_KEY="your_fred_api_key_here"
```

### 3. Dependencies

The FRED adapter requires `aiohttp` for HTTP client functionality:
```bash
pip install aiohttp>=3.9.0
```

## Quick Start

### Basic Usage

```python
import asyncio
from nautilus_trader.adapters.fred import (
    FREDDataClient,
    FREDInstrumentProvider,
    FREDDataClientConfig,
    EconomicData
)
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId, InstrumentId, Symbol, Venue

# Create instrument provider
instrument_provider = FREDInstrumentProvider(
    config=FREDInstrumentProviderConfig(
        api_key="your_api_key",
        series_ids=["GDP", "UNRATE", "FEDFUNDS"]
    )
)

# Load instruments
await instrument_provider.load_all_async()

# Create data client
client = FREDDataClient(
    loop=loop,
    msgbus=msgbus,
    cache=cache,
    clock=clock,
    instrument_provider=instrument_provider,
    config=FREDDataClientConfig(api_key="your_api_key"),
)

# Subscribe to economic data
gdp_instrument = InstrumentId(Symbol("GDP"), Venue("FRED"))
data_type = DataType(EconomicData, metadata={"instrument_id": gdp_instrument})
await client.subscribe_data(data_type)
```

### Strategy Integration

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.adapters.fred.data import EconomicData

class EconomicAwareStrategy(Strategy):
    def on_start(self):
        # Subscribe to economic indicators
        gdp_data_type = DataType(
            EconomicData,
            metadata={
                "instrument_id": InstrumentId(Symbol("GDP"), Venue("FRED")),
                "update_interval": 3600
            }
        )
        self.subscribe_data(gdp_data_type, ClientId("FRED"))
    
    def on_data(self, data):
        if isinstance(data, EconomicData):
            self.process_economic_signal(data)
            
    def process_economic_signal(self, data: EconomicData):
        if data.series_id == "GDP":
            gdp_value = float(data.value)
            if gdp_value > 25000:  # Example threshold
                self.log.info(f"Strong GDP growth: ${gdp_value:.1f}B")
                # Implement trading logic based on GDP
```

## Configuration

### FREDDataClientConfig

```python
config = FREDDataClientConfig(
    api_key="your_fred_api_key",
    base_url="https://api.stlouisfed.org/fred",  # Default
    request_timeout=30.0,
    rate_limit_delay=1.0,  # Seconds between requests
    max_retries=3,
    retry_delay=2.0,
    default_limit=1000,
    update_interval=3600,  # Seconds
    series_ids=["GDP", "UNRATE", "FEDFUNDS"],
    auto_subscribe=False,
)
```

### FREDInstrumentProviderConfig

```python
config = FREDInstrumentProviderConfig(
    api_key="your_fred_api_key",
    series_ids=["GDP", "UNRATE", "CPIAUCSL"],
    load_all=False,  # Don't load all 800k+ series
    category_ids=[1, 32],  # Load from specific categories
    search_terms=["GDP", "unemployment"],
    max_search_results=100,
    cache_instruments=True,
)
```

## Popular Economic Indicators

The adapter provides easy access to popular economic indicators:

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| GDP | Gross Domestic Product | Quarterly |
| UNRATE | Unemployment Rate | Monthly |
| CPIAUCSL | Consumer Price Index | Monthly |
| FEDFUNDS | Federal Funds Rate | Monthly |
| DGS10 | 10-Year Treasury Rate | Daily |
| M1SL | M1 Money Supply | Monthly |
| HOUST | Housing Starts | Monthly |
| INDPRO | Industrial Production | Monthly |

## Data Types

### EconomicData

The `EconomicData` class represents economic time series data:

```python
@customdataclass
class EconomicData(Data):
    instrument_id: InstrumentId
    series_id: str
    value: Decimal
    units: str
    frequency: str
    seasonal_adjustment: str
    last_updated: int
    release_date: int
```

Example usage:
```python
data = EconomicData.create(
    instrument_id=InstrumentId(Symbol("GDP"), Venue("FRED")),
    series_id="GDP",
    value="25000.5",
    units="Billions of Dollars",
    frequency="Quarterly",
    seasonal_adjustment="Seasonally Adjusted Annual Rate",
    ts_event=timestamp,
    ts_init=timestamp,
)

print(f"GDP Value: {data.value} {data.units}")
print(f"Valid: {data.is_valid_value}")
```

## Advanced Usage

### Historical Data Requests

```python
# Request historical GDP data
gdp_instrument = InstrumentId(Symbol("GDP"), Venue("FRED"))
data_type = DataType(
    EconomicData,
    metadata={
        "instrument_id": gdp_instrument,
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "limit": 100,
        "sort_order": "desc"
    }
)

await client.request_data(data_type, correlation_id)
```

### Multi-Series Subscriptions

```python
# Subscribe to multiple economic indicators
indicators = ["GDP", "UNRATE", "FEDFUNDS", "CPIAUCSL"]

for series_id in indicators:
    instrument_id = InstrumentId(Symbol(series_id), Venue("FRED"))
    data_type = DataType(EconomicData, metadata={"instrument_id": instrument_id})
    await client.subscribe_data(data_type)
```

### Category-Based Loading

```python
# Load all series from employment categories
provider_config = FREDInstrumentProviderConfig(
    api_key="your_api_key",
    category_ids=[10, 11, 12],  # Employment categories
    max_search_results=50
)

provider = FREDInstrumentProvider(config=provider_config)
await provider.load_all_async()
```

## Integration with Trading Node

```python
from nautilus_trader.live.node import TradingNode
from nautilus_trader.config import TradingNodeConfig

# Configure FRED data client
config = TradingNodeConfig(
    data_clients={
        "FRED": {
            "api_key": "your_fred_api_key",
            "series_ids": ["GDP", "UNRATE", "FEDFUNDS"],
            "auto_subscribe": True,
            "update_interval": 3600,
        }
    },
    instrument_providers={
        "FRED": {
            "api_key": "your_fred_api_key",
            "series_ids": ["GDP", "UNRATE", "FEDFUNDS"],
        }
    },
    strategies=[
        {
            "strategy_path": "strategies.economic_strategy:EconomicAwareStrategy",
            "config": {"symbol": "EURUSD"}
        }
    ]
)

# Create and start trading node
node = TradingNode(config=config)
await node.run_async()
```

## Best Practices

### Rate Limiting

FRED allows 120 requests per 60 seconds. The adapter implements rate limiting:
- Default delay: 1 second between requests
- Automatic backoff on rate limit exceeded (429 errors)
- Configurable rate limiting parameters

### Caching

- Enable instrument caching to reduce API calls
- Use reasonable update intervals (economic data updates infrequently)
- Cache frequently accessed historical data

### Error Handling

```python
try:
    await client.connect()
except ValueError as e:
    logger.error(f"FRED API configuration error: {e}")
except Exception as e:
    logger.error(f"FRED API connection error: {e}")
```

### Data Validation

```python
def validate_economic_data(data: EconomicData) -> bool:
    if not data.is_valid_value:
        logger.warning(f"Invalid economic data for {data.series_id}")
        return False
    
    # Check for reasonable value ranges
    if data.series_id == "UNRATE" and float(data.value) > 30:
        logger.warning(f"Unusually high unemployment rate: {data.value}%")
        return False
    
    return True
```

## Limitations

### API Constraints

- **Rate Limits**: 120 requests per 60 seconds
- **Data Lag**: Economic data has publication delays (monthly/quarterly)
- **Historical Limits**: Some series have limited historical data
- **Revisions**: Economic data may be revised in subsequent releases

### Data Characteristics

- **Low Frequency**: Most economic data is monthly or quarterly
- **Publication Delays**: GDP data is released with ~1 month delay
- **Seasonal Effects**: Many series require seasonal adjustment consideration
- **Revisions**: Historical data points may change in later releases

## Testing

### Unit Tests

Run the test suite:
```bash
pytest tests/unit_tests/adapters/fred/ -v
```

### Integration Tests

Test with real FRED API (requires API key):
```bash
export FRED_API_KEY="your_test_key"
pytest tests/integration_tests/adapters/fred/ -v
```

### Mock Testing

Use mock responses for development:
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fred_client_with_mock():
    with patch('nautilus_trader.adapters.fred.http.FREDHttpClient') as mock_client:
        mock_client.get_series_observations.return_value = {
            "observations": [
                {"date": "2023-01-01", "value": "25000.0"}
            ]
        }
        # Test your client logic
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   ValueError: FRED API key is required
   ```
   Solution: Set `FRED_API_KEY` environment variable or provide in config.

2. **Rate Limiting**
   ```
   FRED API rate limit exceeded, backing off
   ```
   Solution: Increase `rate_limit_delay` in configuration.

3. **Invalid Series ID**
   ```
   No data found for FRED series: INVALID_ID
   ```
   Solution: Verify series ID exists on FRED website.

4. **Network Timeouts**
   ```
   FRED API request timeout
   ```
   Solution: Increase `request_timeout` in configuration.

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger('nautilus_trader.adapters.fred').setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor API usage:
```python
# Log API call metrics
@strategy.on_data
def on_economic_data(self, data: EconomicData):
    self.log.info(f"Received {data.series_id}: {data.value} at {data.ts_event}")
```

## Examples

Complete examples are available in:
- `examples/strategies/fred_economic_strategy.py` - Economic data integration strategy
- `tests/integration_tests/adapters/fred/` - Integration test examples

## Support

For issues and questions:
- Check the [FRED API documentation](https://fred.stlouisfed.org/docs/api/)
- Review the NautilusTrader adapter development guide
- Submit issues to the NautilusTrader GitHub repository

## License

This FRED adapter is provided under the GNU Lesser General Public License Version 3.0, consistent with the NautilusTrader project licensing.