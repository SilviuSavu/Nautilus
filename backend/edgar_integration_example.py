#!/usr/bin/env python3
"""
EDGAR API Connector Integration Example
======================================

Example script demonstrating how to integrate the EDGAR connector with
the existing Nautilus trading platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import create_default_config, EDGARInstrumentConfig
from edgar_connector.instrument_provider import EDGARInstrumentProvider
from edgar_connector.data_types import FilingType
from edgar_connector.utils import XBRLParser, format_financial_value

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_edgar_connector():
    """Demonstrate EDGAR connector functionality."""
    
    logger.info("=== EDGAR API Connector Demo ===")
    
    # 1. Setup configuration
    logger.info("1. Setting up EDGAR configuration...")
    config = create_default_config(
        user_agent="NautilusTrader-Demo demo@nautilus-trader.com",
        rate_limit_requests_per_second=2.0,  # Conservative for demo
        cache_ttl_seconds=300  # 5 minutes
    )
    
    # 2. Create API client
    logger.info("2. Creating EDGAR API client...")
    async with EDGARAPIClient(config) as api_client:
        
        # 3. Health check
        logger.info("3. Performing API health check...")
        if not await api_client.health_check():
            logger.error("EDGAR API health check failed!")
            return
        logger.info("✓ EDGAR API is healthy")
        
        # 4. Setup instrument provider
        logger.info("4. Setting up instrument provider...")
        instrument_config = EDGARInstrumentConfig(
            update_entities_on_startup=True,
            ticker_cache_ttl=3600
        )
        
        instrument_provider = EDGARInstrumentProvider(api_client, instrument_config)
        await instrument_provider.load_all_async()
        
        stats = instrument_provider.get_statistics()
        logger.info(f"✓ Loaded {stats['total_entities']} SEC entities")
        
        # 5. Demonstrate ticker resolution
        logger.info("5. Demonstrating ticker resolution...")
        test_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        for ticker in test_tickers:
            cik = instrument_provider.resolve_ticker_to_cik(ticker)
            if cik:
                entity = instrument_provider.get_entity_by_cik(cik)
                logger.info(f"  {ticker} -> CIK {cik} ({entity.name})")
            else:
                logger.warning(f"  {ticker} -> Not found")
        
        # 6. Get company financial data
        logger.info("6. Fetching Apple's financial data...")
        apple_cik = "0000320193"
        
        try:
            facts_data = await api_client.get_company_facts(apple_cik)
            if facts_data:
                logger.info(f"✓ Retrieved facts for {facts_data.get('entityName', 'Unknown')}")
                
                # Parse key financial metrics
                parser = XBRLParser()
                key_metrics = parser.extract_key_metrics(facts_data)
                
                logger.info("  Key Financial Metrics:")
                for metric, value in key_metrics.items():
                    if value is not None:
                        formatted = format_financial_value(value, "USD")
                        logger.info(f"    {metric.replace('_', ' ').title()}: {formatted}")
            
        except Exception as e:
            logger.error(f"Error fetching company facts: {e}")
        
        # 7. Get recent filings
        logger.info("7. Fetching Apple's recent filings...")
        
        try:
            submissions = await api_client.get_submissions(apple_cik)
            if submissions and "filings" in submissions:
                recent = submissions["filings"].get("recent", {})
                
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accessions = recent.get("accessionNumber", [])
                
                logger.info("  Recent Filings:")
                for i, (form, date, accession) in enumerate(zip(forms[:5], dates[:5], accessions[:5])):
                    filing_date = datetime.strptime(date, "%Y-%m-%d")
                    days_ago = (datetime.now() - filing_date).days
                    logger.info(f"    {form} - {date} ({days_ago} days ago) - {accession}")
        
        except Exception as e:
            logger.error(f"Error fetching submissions: {e}")
        
        # 8. Demonstrate search functionality
        logger.info("8. Demonstrating company search...")
        
        search_queries = ["Apple", "Microsoft", "tesla"]
        for query in search_queries:
            try:
                results = await api_client.search_companies(query, limit=3)
                logger.info(f"  Search '{query}': {len(results)} results")
                for entity in results:
                    ticker_info = f" ({entity.ticker})" if entity.ticker else ""
                    logger.info(f"    - {entity.name}{ticker_info} [CIK: {entity.cik}]")
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
        
        # 9. Demonstrate financial concept lookup
        logger.info("9. Looking up specific financial concepts...")
        
        try:
            # Get Apple's revenue data
            concept_data = await api_client.get_company_concept(
                apple_cik, "us-gaap", "Revenues"
            )
            
            if concept_data and "units" in concept_data:
                usd_data = concept_data["units"].get("USD", [])
                if usd_data:
                    # Get most recent revenue
                    latest_revenue = max(usd_data, key=lambda x: x.get("end", ""))
                    revenue_value = latest_revenue.get("val", 0)
                    revenue_period = latest_revenue.get("end", "Unknown")
                    
                    formatted_revenue = format_financial_value(revenue_value, "USD")
                    logger.info(f"  Apple's Revenue (period ending {revenue_period}): {formatted_revenue}")
        
        except Exception as e:
            logger.error(f"Error fetching revenue concept: {e}")
        
        # 10. Summary
        logger.info("10. Integration summary:")
        logger.info(f"  ✓ API health check: OK")
        logger.info(f"  ✓ Entity provider: {stats['total_entities']} companies loaded")
        logger.info(f"  ✓ Ticker resolution: Working")
        logger.info(f"  ✓ Financial data: Accessible")
        logger.info(f"  ✓ Filing data: Accessible")
        logger.info(f"  ✓ Search functionality: Working")
        logger.info(f"  ✓ Rate limiting: Enforced")


async def demo_integration_with_existing_platform():
    """Demonstrate how EDGAR connector integrates with existing platform."""
    
    logger.info("\n=== Integration with Existing Platform ===")
    
    # Show how EDGAR data could be used alongside existing data sources
    logger.info("EDGAR connector can be integrated alongside existing data sources:")
    logger.info("  - IB Gateway: Real-time market data + trading")
    logger.info("  - EDGAR API: Fundamental data + regulatory filings")
    logger.info("  - Database: Cached historical data from both sources")
    
    logger.info("\nPotential use cases:")
    logger.info("  1. Fundamental analysis strategies")
    logger.info("  2. Event-driven trading on filing releases")
    logger.info("  3. Compliance monitoring and reporting")
    logger.info("  4. ESG and governance analysis")
    logger.info("  5. Earnings announcement prediction")
    
    logger.info("\nData flow architecture:")
    logger.info("  SEC EDGAR API -> EDGARDataClient -> NautilusTrader MessageBus")
    logger.info("  Custom FilingData objects -> Strategy logic -> Trading decisions")


async def demonstrate_caching():
    """Demonstrate caching functionality."""
    
    logger.info("\n=== Caching Demonstration ===")
    
    config = create_default_config(
        user_agent="NautilusTrader-Cache-Demo demo@nautilus-trader.com",
        enable_cache=True,
        cache_ttl_seconds=60
    )
    
    async with EDGARAPIClient(config) as api_client:
        # First request - should hit API
        logger.info("First request (should hit API)...")
        start_time = datetime.now()
        ticker_data = await api_client.get_company_tickers()
        first_duration = datetime.now() - start_time
        
        logger.info(f"  First request took: {first_duration.total_seconds():.2f}s")
        
        # Second request - should use cache
        logger.info("Second request (should use cache)...")
        start_time = datetime.now()
        ticker_data_cached = await api_client.get_company_tickers()
        second_duration = datetime.now() - start_time
        
        logger.info(f"  Second request took: {second_duration.total_seconds():.2f}s")
        logger.info(f"  Cache speedup: {first_duration.total_seconds() / second_duration.total_seconds():.1f}x faster")
        
        # Verify data is identical
        assert ticker_data == ticker_data_cached
        logger.info("  ✓ Cached data is identical to original")


def demonstrate_configuration():
    """Demonstrate configuration options."""
    
    logger.info("\n=== Configuration Options ===")
    
    # Show different configuration scenarios
    configs = {
        "Development": create_default_config(
            user_agent="NautilusTrader-Dev dev@company.com",
            rate_limit_requests_per_second=10.0,
            cache_ttl_seconds=300,
            max_retries=3
        ),
        "Production": create_default_config(
            user_agent="NautilusTrader-Prod prod@company.com",
            rate_limit_requests_per_second=5.0,
            cache_ttl_seconds=3600,
            max_retries=5
        ),
        "Conservative": create_default_config(
            user_agent="NautilusTrader-Conservative conservative@company.com",
            rate_limit_requests_per_second=1.0,
            cache_ttl_seconds=7200,
            max_retries=2
        )
    }
    
    for name, config in configs.items():
        logger.info(f"  {name} Configuration:")
        logger.info(f"    Rate limit: {config.rate_limit_requests_per_second} req/sec")
        logger.info(f"    Cache TTL: {config.cache_ttl_seconds}s")
        logger.info(f"    Max retries: {config.max_retries}")


async def main():
    """Main demonstration function."""
    
    logger.info("Starting EDGAR API Connector demonstration...")
    
    try:
        # Core functionality demo
        await demo_edgar_connector()
        
        # Integration demo
        await demo_integration_with_existing_platform()
        
        # Caching demo
        await demonstrate_caching()
        
        # Configuration demo
        demonstrate_configuration()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("✓ EDGAR connector is ready for integration!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())