"""
Interactive Brokers Asset Class Support
Comprehensive support for multiple asset classes including stocks, options, futures, forex, bonds, and more.
"""

import logging
from typing import Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum

from ibapi.contract import Contract, ContractDetails
from ib_instrument_provider import IBContractRequest, IBInstrument


class IBAssetClass(Enum):
    """IB Asset Classes"""
    STOCK = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"
    FOREX = "CASH"
    INDEX = "IND"
    BOND = "BOND"
    COMMODITY = "CMDTY"
    WARRANT = "WAR"
    FUND = "FUND"
    CFD = "CFD"
    CRYPTO = "CRYPTO"


class IBOptionRight(Enum):
    """Option Rights"""
    CALL = "C"
    PUT = "P"


class IBExchange(Enum):
    """Major IB Exchanges by Asset Class"""
    # Stocks
    SMART = "SMART"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    BATS = "BATS"
    
    # Options
    CBOE = "CBOE"
    ISE = "ISE"
    PHLX = "PHLX"
    
    # Futures
    CME = "CME"
    NYMEX = "NYMEX"
    GLOBEX = "GLOBEX"
    CBOT = "CBOT"
    ICE = "ICE"
    
    # Forex
    IDEALPRO = "IDEALPRO"
    
    # International
    LSE = "LSE"  # London Stock Exchange
    TSE = "TSE"  # Tokyo Stock Exchange
    SEHK = "SEHK"  # Hong Kong Stock Exchange
    ASX = "ASX"  # Australian Securities Exchange
    FWB = "FWB"  # Frankfurt Stock Exchange


@dataclass
class IBStockContract:
    """Stock contract specification"""
    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str | None = None
    local_symbol: str | None = None
    trading_class: str | None = None


@dataclass
class IBOptionContract:
    """Option contract specification"""
    symbol: str
    expiry: str  # YYYYMMDD format
    strike: float
    right: str  # 'C' or 'P'
    exchange: str = "SMART"
    currency: str = "USD"
    multiplier: str = "100"
    trading_class: str | None = None
    local_symbol: str | None = None


@dataclass
class IBFutureContract:
    """Future contract specification"""
    symbol: str
    expiry: str  # YYYYMMDD format
    exchange: str
    currency: str = "USD"
    multiplier: str | None = None
    trading_class: str | None = None
    local_symbol: str | None = None
    include_expired: bool = False


@dataclass
class IBForexContract:
    """Forex contract specification"""
    symbol: str  # Base currency (e.g., EUR)
    currency: str  # Quote currency (e.g., USD)
    exchange: str = "IDEALPRO"
    local_symbol: str | None = None


@dataclass
class IBBondContract:
    """Bond contract specification"""
    symbol: str
    exchange: str
    currency: str = "USD"
    sec_id_type: str | None = None  # CUSIP, ISIN, etc.
    sec_id: str | None = None


@dataclass
class IBIndexContract:
    """Index contract specification"""
    symbol: str
    exchange: str
    currency: str = "USD"


@dataclass
class IBCFDContract:
    """CFD contract specification"""
    symbol: str
    exchange: str
    currency: str = "USD"


class IBAssetClassManager:
    """
    Interactive Brokers Asset Class Manager
    
    Provides comprehensive support for creating contracts across all
    major asset classes supported by Interactive Brokers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Asset class configurations
        self._setup_asset_class_configs()
        
        # Contract builders
        self.contract_builders = {
            IBAssetClass.STOCK: self._build_stock_contract, IBAssetClass.OPTION: self._build_option_contract, IBAssetClass.FUTURE: self._build_future_contract, IBAssetClass.FOREX: self._build_forex_contract, IBAssetClass.BOND: self._build_bond_contract, IBAssetClass.INDEX: self._build_index_contract, IBAssetClass.CFD: self._build_cfd_contract, IBAssetClass.WARRANT: self._build_warrant_contract, IBAssetClass.FUND: self._build_fund_contract
        }
    
    def _setup_asset_class_configs(self):
        """Setup asset class specific configurations"""
        
        # Default exchanges by asset class
        self.default_exchanges = {
            IBAssetClass.STOCK: IBExchange.SMART.value, IBAssetClass.OPTION: IBExchange.SMART.value, IBAssetClass.FUTURE: IBExchange.GLOBEX.value, IBAssetClass.FOREX: IBExchange.IDEALPRO.value, IBAssetClass.BOND: IBExchange.SMART.value, IBAssetClass.INDEX: IBExchange.SMART.value, IBAssetClass.CFD: IBExchange.SMART.value, IBAssetClass.WARRANT: IBExchange.SMART.value, IBAssetClass.FUND: IBExchange.SMART.value
        }
        
        # Common currency pairs for forex
        self.major_forex_pairs = [
            ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "CHF"), ("AUD", "USD"), ("USD", "CAD"), ("NZD", "USD"), ("EUR", "GBP"), ("EUR", "JPY"), ("GBP", "JPY"), ("CHF", "JPY"), ("EUR", "CHF"), ("AUD", "JPY"), ("GBP", "CHF"), ("EUR", "CAD"), ("AUD", "CAD"), ("CAD", "JPY"), ("NZD", "JPY"), ("GBP", "CAD"), ("GBP", "AUD")
        ]
        
        # Popular futures by exchange
        self.popular_futures = {
            IBExchange.CME.value: ["ES", "NQ", "YM", "RTY"], # E-mini S&P, Nasdaq, Dow, Russell
            IBExchange.CBOT.value: ["ZN", "ZB", "ZF", "ZT"], # Treasury futures
            IBExchange.NYMEX.value: ["CL", "NG", "GC", "SI"], # Oil, Gas, Gold, Silver
            IBExchange.GLOBEX.value: ["6E", "6B", "6J", "6A"]  # Currency futures
        }
        
        # Option expiration patterns
        self.option_expiration_patterns = {
            "monthly": "Third Friday of each month", "weekly": "Every Friday", "quarterly": "Third Friday of March, June, September, December"
        }
    
    def create_stock_contract(self, spec: IBStockContract) -> Contract:
        """Create stock contract"""
        return self._build_stock_contract(spec.__dict__)
    
    def create_option_contract(self, spec: IBOptionContract) -> Contract:
        """Create option contract"""
        return self._build_option_contract(spec.__dict__)
    
    def create_future_contract(self, spec: IBFutureContract) -> Contract:
        """Create future contract"""
        return self._build_future_contract(spec.__dict__)
    
    def create_forex_contract(self, spec: IBForexContract) -> Contract:
        """Create forex contract"""
        return self._build_forex_contract(spec.__dict__)
    
    def create_bond_contract(self, spec: IBBondContract) -> Contract:
        """Create bond contract"""
        return self._build_bond_contract(spec.__dict__)
    
    def create_index_contract(self, spec: IBIndexContract) -> Contract:
        """Create index contract"""
        return self._build_index_contract(spec.__dict__)
    
    def create_cfd_contract(self, spec: IBCFDContract) -> Contract:
        """Create CFD contract"""
        return self._build_cfd_contract(spec.__dict__)
    
    def _build_stock_contract(self, params: dict[str, Any]) -> Contract:
        """Build stock contract"""
        contract = Contract()
        contract.secType = IBAssetClass.STOCK.value
        contract.symbol = params["symbol"]
        contract.exchange = params.get("exchange", self.default_exchanges[IBAssetClass.STOCK])
        contract.currency = params.get("currency", "USD")
        
        if params.get("primary_exchange"):
            contract.primaryExchange = params["primary_exchange"]
        if params.get("local_symbol"):
            contract.localSymbol = params["local_symbol"]
        if params.get("trading_class"):
            contract.tradingClass = params["trading_class"]
        
        return contract
    
    def _build_option_contract(self, params: dict[str, Any]) -> Contract:
        """Build option contract"""
        contract = Contract()
        contract.secType = IBAssetClass.OPTION.value
        contract.symbol = params["symbol"]
        contract.exchange = params.get("exchange", self.default_exchanges[IBAssetClass.OPTION])
        contract.currency = params.get("currency", "USD")
        contract.lastTradeDateOrContractMonth = params["expiry"]
        contract.strike = float(params["strike"])
        contract.right = params["right"]
        contract.multiplier = params.get("multiplier", "100")
        
        if params.get("trading_class"):
            contract.tradingClass = params["trading_class"]
        if params.get("local_symbol"):
            contract.localSymbol = params["local_symbol"]
        
        return contract
    
    def _build_future_contract(self, params: dict[str, Any]) -> Contract:
        """Build future contract"""
        contract = Contract()
        contract.secType = IBAssetClass.FUTURE.value
        contract.symbol = params["symbol"]
        contract.exchange = params["exchange"]
        contract.currency = params.get("currency", "USD")
        contract.lastTradeDateOrContractMonth = params["expiry"]
        
        if params.get("multiplier"):
            contract.multiplier = params["multiplier"]
        if params.get("trading_class"):
            contract.tradingClass = params["trading_class"]
        if params.get("local_symbol"):
            contract.localSymbol = params["local_symbol"]
        if params.get("include_expired"):
            contract.includeExpired = params["include_expired"]
        
        return contract
    
    def _build_forex_contract(self, params: dict[str, Any]) -> Contract:
        """Build forex contract"""
        contract = Contract()
        contract.secType = IBAssetClass.FOREX.value
        contract.symbol = params["symbol"]  # Base currency
        contract.currency = params["currency"]  # Quote currency
        contract.exchange = params.get("exchange", self.default_exchanges[IBAssetClass.FOREX])
        
        if params.get("local_symbol"):
            contract.localSymbol = params["local_symbol"]
        
        return contract
    
    def _build_bond_contract(self, params: dict[str, Any]) -> Contract:
        """Build bond contract"""
        contract = Contract()
        contract.secType = IBAssetClass.BOND.value
        contract.symbol = params["symbol"]
        contract.exchange = params["exchange"]
        contract.currency = params.get("currency", "USD")
        
        if params.get("sec_id_type") and params.get("sec_id"):
            contract.secIdType = params["sec_id_type"]
            contract.secId = params["sec_id"]
        
        return contract
    
    def _build_index_contract(self, params: dict[str, Any]) -> Contract:
        """Build index contract"""
        contract = Contract()
        contract.secType = IBAssetClass.INDEX.value
        contract.symbol = params["symbol"]
        contract.exchange = params["exchange"]
        contract.currency = params.get("currency", "USD")
        
        return contract
    
    def _build_cfd_contract(self, params: dict[str, Any]) -> Contract:
        """Build CFD contract"""
        contract = Contract()
        contract.secType = IBAssetClass.CFD.value
        contract.symbol = params["symbol"]
        contract.exchange = params["exchange"]
        contract.currency = params.get("currency", "USD")
        
        return contract
    
    def _build_warrant_contract(self, params: dict[str, Any]) -> Contract:
        """Build warrant contract"""
        contract = Contract()
        contract.secType = IBAssetClass.WARRANT.value
        contract.symbol = params["symbol"]
        contract.exchange = params.get("exchange", self.default_exchanges[IBAssetClass.WARRANT])
        contract.currency = params.get("currency", "USD")
        
        if params.get("expiry"):
            contract.lastTradeDateOrContractMonth = params["expiry"]
        
        return contract
    
    def _build_fund_contract(self, params: dict[str, Any]) -> Contract:
        """Build fund contract"""
        contract = Contract()
        contract.secType = IBAssetClass.FUND.value
        contract.symbol = params["symbol"]
        contract.exchange = params.get("exchange", self.default_exchanges[IBAssetClass.FUND])
        contract.currency = params.get("currency", "USD")
        
        return contract
    
    def create_contract_from_params(self, asset_class: str, **params) -> Contract:
        """Create contract from asset class and parameters"""
        try:
            asset_class_enum = IBAssetClass(asset_class)
            builder = self.contract_builders.get(asset_class_enum)
            
            if not builder:
                raise ValueError(f"Unsupported asset class: {asset_class}")
            
            return builder(params)
            
        except ValueError as e:
            self.logger.error(f"Invalid asset class: {asset_class}")
            raise
        except Exception as e:
            self.logger.error(f"Error creating contract: {e}")
            raise
    
    def get_supported_asset_classes(self) -> list[str]:
        """Get list of supported asset classes"""
        return [asset_class.value for asset_class in IBAssetClass]
    
    def get_default_exchange(self, asset_class: str) -> str:
        """Get default exchange for asset class"""
        try:
            asset_class_enum = IBAssetClass(asset_class)
            return self.default_exchanges.get(asset_class_enum, IBExchange.SMART.value)
        except ValueError:
            return IBExchange.SMART.value
    
    def get_major_forex_pairs(self) -> list[tuple]:
        """Get major forex currency pairs"""
        return self.major_forex_pairs.copy()
    
    def get_popular_futures(self, exchange: str = None) -> dict[str, list[str]]:
        """Get popular futures by exchange"""
        if exchange:
            return {exchange: self.popular_futures.get(exchange, [])}
        return self.popular_futures.copy()
    
    def validate_option_params(self, symbol: str, expiry: str, strike: float, right: str) -> bool:
        """Validate option parameters"""
        try:
            # Validate expiry format (YYYYMMDD)
            if len(expiry) != 8 or not expiry.isdigit():
                return False
            
            datetime.strptime(expiry, "%Y%m%d")
            
            # Validate strike
            if strike <= 0:
                return False
            
            # Validate right
            if right not in [IBOptionRight.CALL.value, IBOptionRight.PUT.value]:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_future_params(self, symbol: str, expiry: str, exchange: str) -> bool:
        """Validate future parameters"""
        try:
            # Validate expiry format (YYYYMMDD)
            if len(expiry) != 8 or not expiry.isdigit():
                return False
            
            datetime.strptime(expiry, "%Y%m%d")
            
            # Check if exchange supports futures
            futures_exchanges = [IBExchange.CME.value, IBExchange.CBOT.value, IBExchange.NYMEX.value, IBExchange.GLOBEX.value]
            
            return exchange in futures_exchanges
            
        except Exception:
            return False
    
    def validate_forex_params(self, base_currency: str, quote_currency: str) -> bool:
        """Validate forex parameters"""
        # Check currency code format (3 characters)
        if len(base_currency) != 3 or len(quote_currency) != 3:
            return False
        
        # Check if it's a valid currency pair
        return (base_currency, quote_currency) in self.major_forex_pairs
    
    def generate_option_chain_requests(self, underlying_symbol: str, expiry_dates: list[str], strike_range: tuple = None, include_calls: bool = True, include_puts: bool = True) -> list[IBContractRequest]:
        """Generate option chain contract requests"""
        requests = []
        
        for expiry in expiry_dates:
            if not self.validate_option_params(underlying_symbol, expiry, 100.0, "C"):
                continue
            
            # Generate requests for calls and puts
            rights = []
            if include_calls:
                rights.append(IBOptionRight.CALL.value)
            if include_puts:
                rights.append(IBOptionRight.PUT.value)
            
            for right in rights:
                request = IBContractRequest(
                    symbol=underlying_symbol, sec_type=IBAssetClass.OPTION.value, expiry=expiry, right=right
                )
                
                # Add strike range if specified
                if strike_range:
                    # This would need to be handled in the search logic
                    request.context = {"strike_range": strike_range}
                
                requests.append(request)
        
        return requests
    
    def generate_futures_chain_requests(self, underlying_symbol: str, exchange: str, months_ahead: int = 6) -> list[IBContractRequest]:
        """Generate futures chain contract requests"""
        requests = []
        
        # Generate expiry dates for next N months
        current_date = datetime.now()
        
        for i in range(months_ahead):
            # Calculate expiry (third Friday of month)
            year = current_date.year
            month = current_date.month + i
            
            if month > 12:
                year += month // 12
                month = month % 12 or 12
            
            # Find third Friday
            third_friday = self._find_third_friday(year, month)
            expiry = third_friday.strftime("%Y%m%d")
            
            if self.validate_future_params(underlying_symbol, expiry, exchange):
                request = IBContractRequest(
                    symbol=underlying_symbol, sec_type=IBAssetClass.FUTURE.value, exchange=exchange, expiry=expiry
                )
                requests.append(request)
        
        return requests
    
    def _find_third_friday(self, year: int, month: int) -> date:
        """Find third Friday of the month"""
        first_day = date(year, month, 1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday is 14 days later
        return first_friday + timedelta(days=14)


# Global asset class manager instance
_ib_asset_class_manager: IBAssetClassManager | None = None

def get_ib_asset_class_manager() -> IBAssetClassManager:
    """Get or create the IB asset class manager singleton"""
    global _ib_asset_class_manager
    
    if _ib_asset_class_manager is None:
        _ib_asset_class_manager = IBAssetClassManager()
    
    return _ib_asset_class_manager

def reset_ib_asset_class_manager():
    """Reset the asset class manager singleton (for testing)"""
    global _ib_asset_class_manager
    _ib_asset_class_manager = None