#!/usr/bin/env python3
"""
Interactive Brokers Configuration for NautilusTrader - DECOMMISSIONED
DEPRECATED: This configuration is DECOMMISSIONED in favor of the Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)
All IBKR connections should now go through the Enhanced IBKR Keep-Alive MarketData Engine at Port 8800.

REASON FOR DECOMMISSION: 
- Prevents duplicate IBKR connections and Client ID conflicts
- Enhanced IBKR Keep-Alive MarketData Engine provides superior performance and reliability
- Centralized IBKR data distribution through dual messagebus architecture
"""

from decimal import Decimal

from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersExecutionClientConfig
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersInstrumentProviderConfig
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveDataClientFactory
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveExecClientFactory
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode


def create_ib_config():
    """DECOMMISSIONED: Create Interactive Brokers configuration for paper trading
    
    This function is DECOMMISSIONED. All IBKR connections should go through:
    Enhanced IBKR Keep-Alive MarketData Engine at Port 8800
    """
    # DECOMMISSIONED: Return None to prevent IBKR connection conflicts
    print("⚠️ DECOMMISSIONED: ib_config.create_ib_config() - Use Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)")
    return None
    
    # Instrument provider configuration
    instrument_provider = InteractiveBrokersInstrumentProviderConfig(
        build_futures_chain=False,
        build_options_chain=False,
        min_expiry_days=10,
        max_expiry_days=60,
        # Load common instruments for testing
        load_ids=frozenset([
            "AAPL.NASDAQ",
            "MSFT.NASDAQ", 
            "SPY.ARCA",
            "EUR/USD.IDEALPRO",
            "GBP/USD.IDEALPRO",
        ]),
    )
    
    # Data client configuration
    data_client_config = InteractiveBrokersDataClientConfig(
        ibg_host="127.0.0.1",  # IB Gateway host
        ibg_port=4002,         # IB Gateway paper trading port
        ibg_client_id=1,       # Client ID
        use_regular_trading_hours=True,
        instrument_provider=instrument_provider,
    )
    
    # Execution client configuration  
    exec_client_config = InteractiveBrokersExecutionClientConfig(
        ibg_host="127.0.0.1",  # IB Gateway host
        ibg_port=4002,         # IB Gateway paper trading port
        ibg_client_id=1,       # Client ID
        account_id="DU12345",  # Paper trading account ID (update with your actual ID)
        instrument_provider=instrument_provider,
    )
    
    # Trading node configuration
    config_node = TradingNodeConfig(
        trader_id="NAUTILUS-IB-001",
        logging=LoggingConfig(
            log_level="INFO",
            log_component_levels={
                "nautilus_trader.adapters.interactive_brokers": "DEBUG",
            }
        ),
        data_clients={
            IB: data_client_config,
        },
        exec_clients={
            IB: exec_client_config,
        },
        data_engine=LiveDataEngineConfig(
            time_bars_timestamp_on_close=False,
            validate_data_sequence=True,
        ),
        timeout_connection=90.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )
    
    return config_node


def create_trading_node():
    """DECOMMISSIONED: Create and configure trading node with IB adapter
    
    This function is DECOMMISSIONED. All IBKR connections should go through:
    Enhanced IBKR Keep-Alive MarketData Engine at Port 8800
    """
    # DECOMMISSIONED: Return None to prevent IBKR connection conflicts
    print("⚠️ DECOMMISSIONED: ib_config.create_trading_node() - Use Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)")
    return None


if __name__ == "__main__":
    """
    Run this configuration for testing IB connection.
    
    Prerequisites:
    1. TWS or IB Gateway running on port 7497
    2. API permissions enabled in TWS/Gateway
    3. Paper trading account (DU12345 or update account_id)
    """
    try:
        print("Starting NautilusTrader with Interactive Brokers adapter...")
        print("Configuration:")
        print("  - Host: 127.0.0.1")
        print("  - Port: 7497 (Paper Trading)")
        print("  - Client ID: 1")
        print("  - Account: DU12345 (update if different)")
        print("\nMake sure TWS/IB Gateway is running with API enabled!")
        print("Press Ctrl+C to stop...\n")
        
        node = create_trading_node()
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if 'node' in locals():
            node.dispose()