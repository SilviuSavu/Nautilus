# Toraniko Factor Engine Integration

## Overview
The Toraniko Factor Engine has been successfully integrated into the Nautilus trading platform as an additional engine alongside the existing NautilusTrader engine. Toraniko provides multi-factor equity risk modeling capabilities for quantitative and systematic trading.

## What is Toraniko?
Toraniko is a complete implementation of a characteristic factor model similar to Barra and Axioma. It enables:

- **Factor Analysis**: Calculate momentum, value, and size factor scores
- **Risk Modeling**: Estimate factor returns and construct covariance matrices  
- **Portfolio Optimization**: Support optimization with factor exposure constraints
- **High Performance**: Process 10+ years of daily data in under a minute

## Integration Architecture

### Directory Structure
```
backend/
├── engines/
│   └── toraniko/                    # Cloned toraniko repository
│       ├── toraniko/
│       │   ├── model.py            # Core factor model
│       │   ├── styles.py           # Style factor calculations
│       │   └── utils.py            # Utility functions
│       └── requirements.txt
├── factor_engine_service.py        # Service wrapper for toraniko
├── factor_engine_routes.py         # FastAPI routes for factor engine
└── test_factor_engine_integration.py  # Integration tests
```

### Dependencies Added
- **numpy~=1.26.2**: Numerical computations
- **polars~=1.0.0**: Fast dataframe operations

## API Endpoints

The factor engine is available through REST API endpoints at `/api/v1/factor-engine/`:

### Health & Info
- `GET /health` - Health check
- `GET /info` - Engine information and capabilities
- `GET /factors/list` - Available factors

### Factor Calculations
- `POST /factors/momentum` - Calculate momentum factor scores
- `POST /factors/value` - Calculate value factor scores  
- `POST /factors/size` - Calculate size factor scores

### Risk Modeling
- `POST /risk-model/create` - Create complete risk model
- `POST /portfolio/exposures` - Calculate portfolio factor exposures

## Usage Examples

### 1. Calculate Momentum Factor
```python
import httpx

# Sample returns data
data = [
    {"symbol": "AAPL", "date": "2024-01-01", "asset_returns": 0.02},
    {"symbol": "AAPL", "date": "2024-01-02", "asset_returns": -0.01},
    # ... more data
]

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/factor-engine/factors/momentum",
        json={
            "data": data,
            "trailing_days": 252,
            "winsor_factor": 0.01
        }
    )
    momentum_scores = response.json()
```

### 2. Calculate Value Factor
```python
# Fundamental data required
fundamental_data = [
    {
        "symbol": "AAPL", 
        "date": "2024-01-01",
        "book_price": 0.34,     # Book-to-price ratio
        "sales_price": 0.08,    # Sales-to-price ratio
        "cf_price": 0.007       # Cash flow-to-price ratio
    },
    # ... more data
]

response = await client.post(
    "http://localhost:8000/api/v1/factor-engine/factors/value",
    json={"data": fundamental_data}
)
value_scores = response.json()
```

### 3. Create Risk Model
```python
response = await client.post(
    "http://localhost:8000/api/v1/factor-engine/risk-model/create",
    json={
        "universe_symbols": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "universe_size": 3000
    }
)
risk_model = response.json()
```

## Factor Types Available

### Style Factors
1. **Momentum**: Tendency of well-performing assets to continue performing well
2. **Value**: Tendency of "cheap" stocks to outperform "expensive" stocks  
3. **Size**: Tendency of small-cap stocks to outperform large-cap stocks

### Sector Factors
- Basic Materials, Communication Services, Consumer Cyclical
- Consumer Defensive, Energy, Financial Services, Healthcare  
- Industrials, Real Estate, Technology, Utilities

### Market Factor
- Overall market exposure and beta

## Data Requirements

### For Momentum Factor
- Asset returns (daily)
- Symbol and date information

### For Value Factor  
- Book value, sales, cash flow data
- Price information to calculate ratios
- Market capitalization

### For Complete Risk Model
- Asset returns (daily, multi-year history preferred)
- Market capitalization data
- Sector classifications (GICS Level 1 suitable)
- Fundamental data for value calculations

## Integration with Existing Systems

### With Interactive Brokers
The factor engine can use market data from IB Gateway:
- Historical price data for return calculations
- Fundamental data for value factor computation
- Real-time data for live factor monitoring

### With NautilusTrader
Factor analysis can complement NautilusTrader strategies:
- Use factor exposures for portfolio construction
- Risk management based on factor loadings
- Strategy optimization with factor constraints

### Database Integration
Factor scores and risk models can be stored in PostgreSQL:
- Time-series factor data
- Historical risk model components
- Portfolio exposure tracking

## Performance Characteristics

- **Speed**: 10+ years of daily factor returns in under 1 minute (M1 MacBook)
- **Accuracy**: Comparable results to professional risk models like Barra
- **Scalability**: Supports large universes (Russell 3000+)
- **Memory Efficient**: Uses Polars for optimal memory usage

## Testing

### Integration Tests
Run the integration test suite:
```bash
cd backend
python3 test_factor_engine_integration.py
```

### API Tests  
Test the API endpoints:
```bash
# Start the backend first
python3 test_factor_api.py
```

### Unit Tests
Run toraniko's built-in tests:
```bash
cd backend/engines/toraniko
python -m pytest toraniko/tests/
```

## Configuration

### Environment Variables
No additional environment variables required. The factor engine uses the same database and Redis connections as the main application.

### Settings
Factor calculation parameters can be adjusted:
- `trailing_days`: Lookback period for momentum (default: 252)
- `winsor_factor`: Outlier winsorization (default: 0.01)
- `universe_size`: Maximum assets in risk model (default: 3000)

## Monitoring & Logging

The factor engine integrates with the existing monitoring system:
- Service health checks via `/health` endpoint
- Detailed logging of factor calculations
- Error handling with appropriate HTTP status codes
- Performance metrics tracking

## Future Enhancements

### Planned Features
1. **Custom Factors**: Support for user-defined factors
2. **Real-time Updates**: Live factor score updates
3. **Advanced Analytics**: Factor attribution analysis
4. **Visualization**: Factor exposure charts and risk decomposition
5. **Backtesting**: Factor-based strategy backtesting

### Data Source Integration
1. **EDGAR Integration**: Fundamental data from SEC filings
2. **Alternative Data**: ESG, sentiment, and macro factors
3. **International Markets**: Multi-region factor models

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure numpy and polars are installed
2. **Missing Data**: Verify required columns in input data
3. **Performance**: Use appropriate date ranges for large datasets

### Support
- Check the toraniko repository: https://github.com/0xfdf/toraniko
- Review integration tests for usage examples
- Consult the API documentation at `/docs` when server is running

## License & Credits

- **Toraniko**: MIT License, created by 0xfdf
- **Integration**: Part of the Nautilus trading platform
- **Repository**: https://github.com/0xfdf/toraniko.git

The integration maintains the original toraniko functionality while providing a seamless interface within the Nautilus platform architecture.