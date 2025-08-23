# DBnomics Adapter for NautilusTrader

The DBnomics adapter provides access to economic and statistical time series data from the [dbnomics.world](https://db.nomics.world/) API. This adapter enables integration of macroeconomic data from various statistical offices, central banks, and international organizations into NautilusTrader strategies.

## Overview

DBnomics is a comprehensive database of economic and statistical data from over 80 official providers including:

- **IMF** (International Monetary Fund) - Global economic indicators
- **OECD** (Organisation for Economic Co-operation and Development) - Economic statistics
- **ECB** (European Central Bank) - Eurozone monetary data  
- **EUROSTAT** - European Union statistics
- **BIS** (Bank for International Settlements) - Banking and financial data
- **World Bank** - Development indicators
- **National statistical offices** worldwide

## Features

- **Data Integration**: Seamlessly fetch economic time series data
- **Flexible Filtering**: Search by provider, dataset, series, or dimensions
- **Automatic Retry**: Built-in retry logic for robust API interactions
- **Error Handling**: Comprehensive error handling for network and data issues
- **Caching Support**: Leverage existing NautilusTrader caching mechanisms
- **Custom Mappings**: Map DBnomics series to custom instrument identifiers

## Installation

The adapter requires the `dbnomics` Python client:

```bash
pip install dbnomics
```

## Configuration

### Basic Configuration

```python
from nautilus_trader.adapters.dbnomics import DBnomicsDataClientConfig

config = DBnomicsDataClientConfig()
```

### Advanced Configuration

```python
config = DBnomicsDataClientConfig(
    max_nb_series=100,  # Maximum series per request
    timeout=30,  # Request timeout in seconds
    default_providers=['IMF', 'OECD', 'ECB'],  # Default data providers
    instrument_id_mappings={
        'IMF/CPI/A.FR.PCPIEC_WT': 'EUR_INFLATION_ANNUAL',
        'OECD/QNA/DEU.B1_GE.GPSA.Q': 'DEU_GDP_QUARTERLY'
    },
    auto_retry=True,  # Enable automatic retries
)
```

## Usage Examples

### 1. Basic Data Client Setup

```python
from nautilus_trader.adapters.dbnomics import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics import DBnomicsDataClientConfig

# Create configuration
config = DBnomicsDataClientConfig(
    max_nb_series=50,
    timeout=30,
)

# Create data client (typically done via engine configuration)
client = DBnomicsDataClient(
    loop=loop,
    client_id=ClientId("DBNOMICS-001"),
    config=config,
    msgbus=msgbus,
    cache=cache,
    clock=clock,
)
```

### 2. Instrument Provider Usage

```python
from nautilus_trader.adapters.dbnomics import DBnomicsInstrumentProvider

# Create provider
provider = DBnomicsInstrumentProvider(
    max_nb_series=50,
    timeout=30,
)

# Load instruments with filters
filters = {
    'providers': ['IMF', 'OECD'],
    'datasets': {
        'IMF': ['CPI', 'IFS'],
        'OECD': ['QNA', 'KEI']
    },
    'dimensions': {
        'geo': ['FR', 'DE', 'US'],  # Countries
        'freq': ['M', 'Q']  # Monthly and Quarterly
    }
}

await provider.load_all_async(filters)

# Get loaded instruments
instruments = provider.get_all()
```

### 3. Data Subscription

```python
from nautilus_trader.adapters.dbnomics import DBnomicsTimeSeriesData
from nautilus_trader.data.messages import DataType, SubscribeData
from nautilus_trader.model.identifiers import InstrumentId, Symbol

# Define the data type and instrument
instrument_id = InstrumentId(
    Symbol("IMF-CPI-A.FR.PCPIEC_WT"), 
    DBNOMICS_VENUE
)

data_type = DataType(
    type=DBnomicsTimeSeriesData,
    metadata={
        'instrument_id': instrument_id,
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'filters': [
            {
                'code': 'interpolate',
                'parameters': {
                    'frequency': 'monthly',
                    'method': 'linear'
                }
            }
        ]
    }
)

# Create subscription command
command = SubscribeData(
    client_id=client_id,
    venue=DBNOMICS_VENUE,
    data_type=data_type,
    command_id=UUID4(),
)

# Subscribe to data
await client._subscribe(command)
```

## Data Types

### DBnomicsTimeSeriesData

The primary data type representing a single time series observation:

```python
class DBnomicsTimeSeriesData(Data):
    instrument_id: InstrumentId  # NautilusTrader instrument identifier
    timestamp: pd.Timestamp      # Data timestamp
    value: Decimal               # Series value
    series_code: str             # Original DBnomics series code
    provider_code: str           # Data provider (e.g., 'IMF')
    dataset_code: str            # Dataset code (e.g., 'CPI')
    frequency: str | None        # Data frequency ('M', 'Q', 'A', etc.)
    unit: str | None             # Unit of measurement
```

## Series Identification

### DBnomics Series Format
DBnomics uses the format: `provider_code/dataset_code/series_code`

Examples:
- `IMF/CPI/A.FR.PCPIEC_WT` - IMF Consumer Price Index for France
- `OECD/QNA/DEU.B1_GE.GPSA.Q` - OECD Quarterly National Accounts for Germany
- `ECB/IRS/M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA` - ECB Interest Rate Statistics

### Instrument Mapping
The adapter converts DBnomics series to NautilusTrader instruments:

- **Series**: `IMF/CPI/A.FR.PCPIEC_WT`
- **Symbol**: `IMF-CPI-A.FR.PCPIEC_WT` (slashes → hyphens)
- **Venue**: `DBNOMICS`
- **InstrumentId**: `IMF-CPI-A.FR.PCPIEC_WT.DBNOMICS`

## Common Use Cases

### 1. Economic Indicators for Strategy Context

```python
# Subscribe to key economic indicators
indicators = [
    "IMF-CPI-A.US.PCPIEC_WT",    # US Inflation
    "OECD-KEI-M.USA.LORSGPRT.STSA",  # US Unemployment
    "IMF-IFS-M.US.FPOLM_PA",     # US Fed Funds Rate
]

for indicator in indicators:
    instrument_id = InstrumentId(Symbol(indicator), DBNOMICS_VENUE)
    # Subscribe to data...
```

### 2. Multi-Country Comparison

```python
# Compare GDP growth across countries
gdp_filters = {
    'providers': ['OECD'],
    'datasets': {'OECD': ['QNA']},
    'dimensions': {
        'geo': ['USA', 'DEU', 'FRA', 'JPN'],
        'subject': ['B1_GE'],  # GDP
        'measure': ['GPSA']    # Growth rate
    }
}

await provider.load_all_async(gdp_filters)
```

### 3. Yield Curve Analysis

```python
# Load yield curve data
yield_filters = {
    'providers': ['OECD'],
    'datasets': {'OECD': ['KEI']},
    'dimensions': {
        'geo': ['USA'],
        'subject': ['IRLT']  # Long-term interest rates
    }
}
```

## Error Handling

The adapter provides comprehensive error handling:

```python
from nautilus_trader.adapters.dbnomics.errors import (
    DBnomicsConnectionError,
    DBnomicsDataError,
    DBnomicsRateLimitError,
)

try:
    await client._connect()
except DBnomicsConnectionError as e:
    logger.error(f"Connection failed: {e}")
except DBnomicsRateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
except DBnomicsDataError as e:
    logger.error(f"Data error: {e}")
```

## Rate Limiting

The adapter leverages the built-in retry mechanisms of the `dbnomics` client:

- **Automatic Retries**: Uses `tenacity` for exponential backoff
- **Respectful Requests**: Follows API rate limits
- **Timeout Handling**: Configurable request timeouts

## Testing

Run the test suite:

```bash
# Unit tests
pytest nautilus_trader/adapters/dbnomics/tests/ -v

# Integration tests (requires network)
pytest nautilus_trader/adapters/dbnomics/tests/ -m integration -v

# Specific test file
pytest nautilus_trader/adapters/dbnomics/tests/test_data_client.py -v
```

## Common Providers and Datasets

| Provider | Code | Popular Datasets | Description |
|----------|------|------------------|-------------|
| IMF | IMF | CPI, IFS, WEO | International Monetary Fund |
| OECD | OECD | QNA, KEI, EO | Economic indicators and statistics |
| ECB | ECB | IRS, BSI | European Central Bank data |
| World Bank | WB | WDI | World Development Indicators |
| Eurostat | EUROSTAT | NAMA_10_GDP | European Union statistics |
| BIS | BIS | CBPOL | Central bank policy rates |

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   ```python
   config = DBnomicsDataClientConfig(timeout=60)  # Increase timeout
   ```

2. **Too Many Series**
   ```python
   config = DBnomicsDataClientConfig(max_nb_series=10)  # Reduce batch size
   ```

3. **Invalid Series Code**
   - Check series format: `provider/dataset/series`
   - Use DBnomics website to verify series existence

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('nautilus_trader.adapters.dbnomics').setLevel(logging.DEBUG)
```

## Contributing

When contributing to the DBnomics adapter:

1. **Follow NautilusTrader patterns**: Maintain consistency with existing adapters
2. **Add tests**: Unit tests for all new functionality
3. **Update documentation**: Keep README and docstrings current
4. **Error handling**: Implement proper exception handling
5. **Performance**: Consider rate limits and caching

## References

- [DBnomics Website](https://db.nomics.world/)
- [DBnomics Python Client](https://github.com/dbnomics/dbnomics-python-client)
- [NautilusTrader Documentation](https://docs.nautilustrader.io/)
- [Data Provider List](https://db.nomics.world/providers)